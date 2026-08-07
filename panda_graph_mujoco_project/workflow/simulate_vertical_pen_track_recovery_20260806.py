#!/usr/bin/env python3
"""Simulate graph-set drawing with a fixed Franka Panda and a passive spring-loaded pen.

The original Panda base, desk, and paper are preserved.  The original rigid pen body, physical tip geom, and pen-tip site are preserved
and reparented under a passive spring carriage in a generated MJCF copy.  The
added mechanism consists of:

* a guide block rigidly held by the gripper;
* the existing rigid pen assembly constrained by one passive prismatic joint
  along the pen axis;
* a linear spring and damper on that joint;
* the existing physical pen-tip geom and touch site, unchanged in name and pose.

Panda joints 1-7 move the guide block through the vertical pen-width line graph
waypoints.  The passive spring lets the pen retreat and recover only along its
axis, protecting the paper and tip from small normal-position errors.

The workflow uses one terminal-owned progress bar.  Approach, direct lowering, transfer, and lift motions use closed-loop resolved-rate Cartesian control.  Each complete vertical line
stroke is then streamed as one continuous arc-length-parameterized Cartesian
reference.  The simulator does not stop, settle, or solve endpoint IK at every
resampled vertical-line point. Every point from prepare_vertical_fill_paths remains on the
reference polyline and is crossed in order, while the passive spring regulates
the paper-normal direction.  Physical pen-paper contact remains logged and creates the live viewer trace.
Each stroke uses approach -> direct lower -> continuous vertical-line drawing -> lift.
There is no contact-search phase and no commanded-trajectory fallback. The simulator writes simulated_drawing.png and simulated_contact_ink.png from the same physically contact-backed ink samples, plus simulated_trajectory.png from all executed draw-command samples. Graph-versus-trajectory node and edge-length errors are evaluated from executed draw motion.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

try:
    import mujoco  # type: ignore
except ModuleNotFoundError:  # Allows --help and syntax validation without MuJoCo.
    mujoco = None

BUILD = "panda_vertical_track_recovery_20260806"
GENERATED_SUFFIX = "_spring_guide_runtime_20260805"
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))

LOG_COLUMNS = [
    "time_s", "target_index", "segment_id", "stroke_id", "component_id",
    "point_index", "mode", "phase", "draw_command", "contact_active", "ink_active",
    "ink_run_id", "physical_contact_count", "touch_sensor_force_n",
    "raw_contact_force_n", "filtered_contact_force_n", "sustained_overforce",
    "spring_joint_position_m",
    "spring_compression_m", "spring_force_estimate_n", "desired_spring_position_m",
    "desired_x_m", "desired_y_m", "desired_z_m", "actual_x_m", "actual_y_m",
    "actual_z_m", "desired_guide_x_m", "desired_guide_y_m", "desired_guide_z_m",
    "actual_guide_x_m", "actual_guide_y_m", "actual_guide_z_m",
    "position_error_m", "xy_error_m", "cartesian_reference_error_m",
    "tip_speed_m_s", "overforce_event",
    *[f"q{i}" for i in range(1, 8)],
    *[f"qref{i}" for i in range(1, 8)],
]


@dataclass(frozen=True)
class PoseTarget:
    guide_position: np.ndarray
    tip_position: np.ndarray
    spring_position: float
    stroke_id: int
    component_id: int
    point_index: int
    mode: str
    segment_id: int


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    converged: bool
    position_error_m: float
    orientation_error_rad: float
    iterations: int


@dataclass(frozen=True)
class MotionResult:
    completed: bool
    overforce: bool
    contact: bool
    maximum_force_n: float
    final_tip_error_m: float


@dataclass
class XmlDocument:
    path: Path
    tree: ET.ElementTree
    root: ET.Element
    includes: list[tuple[ET.Element, Path]]


class SingleProgressBar:
    """Exactly one progress line for the complete MuJoCo stage.

    Interactive terminals receive carriage-return updates on the same line.  If
    stdout is redirected, intermediate updates are suppressed and exactly one
    final progress line is emitted.  This prevents captured carriage returns from
    appearing as many separate bars.
    """

    def __init__(self, total: int, *, enabled: bool = True, width: int = 18) -> None:
        self.total = max(1, int(total))
        self.enabled = bool(enabled)
        self.interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.width = max(10, int(width))
        self.completed = 0
        self.started = time.perf_counter()
        self.last_render = 0.0
        self.last_detail = "starting"

    def _text(self) -> str:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        fraction = self.completed / self.total
        filled = min(self.width, int(round(self.width * fraction)))
        # Waypoint density is not proportional to dynamics cost, and startup
        # contact acquisition is much slower than steady drawing.  Displaying an
        # ETA from waypoint rate produced meaningless 100-hour estimates.  Keep
        # the single bar honest by showing elapsed wall time instead.
        elapsed_text = format_duration(elapsed)
        return (
            f"Drawing [{'#' * filled}{'-' * (self.width - filled)}] "
            f"{self.completed}/{self.total} {100.0 * fraction:5.1f}% "
            f"elapsed {elapsed_text} | {self.last_detail[:30]}"
        )

    def update(self, increment: int = 1, detail: str | None = None, *, force: bool = False) -> None:
        self.completed = min(self.total, self.completed + max(0, int(increment)))
        if detail:
            self.last_detail = detail
        if not self.enabled or not self.interactive:
            return
        now = time.perf_counter()
        if not force and self.completed < self.total and now - self.last_render < 0.10:
            return
        sys.stdout.write("\r\033[2K" + self._text())
        sys.stdout.flush()
        self.last_render = now

    def finish(self, detail: str = "complete") -> None:
        if self.completed < self.total:
            self.completed = self.total
        self.last_detail = detail
        if not self.enabled:
            return
        if self.interactive:
            sys.stdout.write("\r\033[2K" + self._text() + "\n")
        else:
            sys.stdout.write(self._text() + "\n")
        sys.stdout.flush()



class ViewerInkRenderer:
    """Persistent black contact trace drawn directly on the MuJoCo paper.

    MuJoCo does not simulate paint deposition by itself.  This renderer adds
    thin capsule geoms to the passive viewer only when the physical pen-tip geom
    is touching the paper during a draw command.  The same condition is logged
    as ``ink_active`` and used for the PNG output.
    """

    def __init__(
        self,
        viewer: Any,
        *,
        paper_z: float,
        radius: float,
        minimum_spacing: float,
        maximum_segments: int,
    ) -> None:
        self.viewer = viewer
        self.paper_z = float(paper_z) + max(5e-5, 0.25 * float(radius))
        self.radius = max(1e-5, float(radius))
        self.minimum_spacing = max(1e-6, float(minimum_spacing))
        self.maximum_segments = max(0, int(maximum_segments))
        self.segment_count = 0
        self.last_stroke_id: int | None = None
        self.last_point: np.ndarray | None = None
        self.enabled = viewer is not None and self.maximum_segments > 0

    def break_stroke(self) -> None:
        self.last_stroke_id = None
        self.last_point = None

    def add_sample(self, stroke_id: int, xyz: np.ndarray) -> None:
        if not self.enabled or self.segment_count >= self.maximum_segments:
            return
        point = np.array([float(xyz[0]), float(xyz[1]), self.paper_z], dtype=float)
        if self.last_stroke_id != int(stroke_id):
            self.last_stroke_id = int(stroke_id)
            self.last_point = point
            return
        assert self.last_point is not None
        distance = float(np.linalg.norm(point[:2] - self.last_point[:2]))
        if distance < self.minimum_spacing:
            return
        try:
            scene = self.viewer.user_scn
            capacity = min(self.maximum_segments, int(scene.maxgeom))
            if int(scene.ngeom) >= capacity:
                self.enabled = False
                return
            geom = scene.geoms[int(scene.ngeom)]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
                np.eye(3, dtype=float).reshape(-1),
                np.array([0.01, 0.01, 0.01, 1.0], dtype=np.float32),
            )
            mujoco.mjv_makeConnector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                self.radius,
                float(self.last_point[0]), float(self.last_point[1]), float(self.last_point[2]),
                float(point[0]), float(point[1]), float(point[2]),
            )
            geom.rgba[:] = np.array([0.01, 0.01, 0.01, 1.0], dtype=np.float32)
            scene.ngeom += 1
            self.segment_count += 1
            self.last_point = point
        except Exception:
            # Viewer decoration must never destabilize the physics simulation.
            self.enabled = False


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Original fixed-base Panda scene with the original desk and paper.")
    parser.add_argument(
        "--spring-model", type=Path, default=None,
        help=(
            "Optional runtime-generated spring MJCF path. When omitted, the model is "
            "written under OUTPUT_DIR/runtime_model so the source model directory "
            "contains only drawing_scene.xml, panda_drawing.xml, and the scene config."
        ),
    )
    parser.add_argument("--rebuild-spring-model", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--strokes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-summary", type=Path, default=None,
                        help="Optional vertical_fill_summary.csv produced by the graph stage.")
    parser.add_argument("--requested-graph-scale", type=int, default=None)
    parser.add_argument("--requested-max-processed-pixels", type=int, default=None)
    parser.add_argument("--micro-edge-max-length-original-px", type=float, default=None)
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument("--viewer", dest="viewer", action="store_true",
                              help="Open the live MuJoCo simulation window (default).")
    viewer_group.add_argument("--no-viewer", dest="viewer", action="store_false",
                              help="Run headless without a MuJoCo window.")
    parser.set_defaults(viewer=True)
    parser.add_argument("--keep-viewer-open", action="store_true")
    parser.add_argument(
        "--viewer-startup-delay", type=float, default=1.50,
        help="Seconds to keep syncing after launch so the WSLg/X11 window becomes visible.",
    )
    parser.add_argument(
        "--viewer-backend", choices=("auto", "x11", "wayland"), default="x11",
        help="GLFW window backend. X11 is the reliable default for WSLg.",
    )
    parser.add_argument(
        "--viewer-software-rendering", action=argparse.BooleanOptionalAction, default=False,
        help="Use Mesa software rendering for the viewer when the WSL GPU path is broken.",
    )
    parser.add_argument("--max-strokes", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable the single terminal drawing progress bar.")

    # Model names and passive spring mechanism.
    parser.add_argument("--pen-tip-site-name", default="pen_tip")
    parser.add_argument("--pen-touch-sensor-name", default="pen_touch")
    parser.add_argument("--paper-geom-name", default="auto")
    parser.add_argument("--guide-site-name", default="spring_pen_guide_site")
    parser.add_argument("--spring-body-name", default="spring_pen_carriage")
    parser.add_argument("--spring-joint-name", default="spring_pen_joint")
    parser.add_argument("--pen-contact-geom-name", default="drawing_pen_tip")
    parser.add_argument("--finger-joint1-name", default="finger_joint1")
    parser.add_argument("--finger-joint2-name", default="finger_joint2")
    parser.add_argument("--gripper-actuator-name", default="actuator8")
    parser.add_argument("--gripper-finger-position", type=float, default=0.005)
    parser.add_argument("--gripper-control", type=float, default=32.0)

    parser.add_argument("--pen-spring-stiffness", type=float, default=40.0,
                        help="Linear pen spring stiffness in N/m.")
    parser.add_argument("--pen-spring-damping", type=float, default=0.50,
                        help="Passive pen-axis damping in N s/m.")
    parser.add_argument("--pen-spring-travel", type=float, default=0.015,
                        help="Maximum compression travel in m.")
    parser.add_argument("--pen-spring-extension", type=float, default=0.002,
                        help="Allowed extension beyond the spring rest position in m.")
    parser.add_argument("--pen-spring-armature", type=float, default=0.001)
    parser.add_argument("--pen-spring-mass", type=float, default=0.055)
    parser.add_argument("--guide-block-half-x", type=float, default=0.008)
    parser.add_argument("--guide-block-half-y", type=float, default=0.008)
    parser.add_argument("--guide-block-half-z", type=float, default=0.018)
    parser.add_argument("--pen-body-radius", type=float, default=0.0025,
                        help="Radius applied to the preserved rigid pen body (m).")
    parser.add_argument("--pen-tip-radius", type=float, default=0.0010,
                        help="Radius applied to the preserved physical tip geom (m).")
    parser.add_argument("--pen-paper-penetration", type=float, default=0.00020)
    parser.add_argument(
        "--guide-press-depth", type=float, default=0.00150,
        help=(
            "Additional fixed downward guide displacement during lower/draw (m). "
            "This is a direct spring preload, not a contact-search phase."
        ),
    )
    parser.add_argument(
        "--lower-contact-gap-tolerance", type=float, default=0.00025,
        help="Numerical pen-bottom gap accepted as seated contact after direct lowering (m).",
    )
    parser.add_argument(
        "--contact-settle-time", type=float, default=0.15,
        help="Time to hold the directly lowered guide before continuous drawing (s).",
    )

    # Motion and contact control.
    parser.add_argument("--draw-speed", type=float, default=0.005)
    parser.add_argument("--transfer-speed", type=float, default=0.030)
    parser.add_argument("--vertical-speed", type=float, default=0.005)
    parser.add_argument(
        "--minimum-segment-time", type=float, default=0.08,
        help="Minimum duration for approach, contact-positioning, and lift motions only; continuous draw strokes do not pay this cost per waypoint.",
    )
    parser.add_argument("--settle-time", type=float, default=0.08)
    parser.add_argument("--max-cartesian-speed", type=float, default=0.08)
    parser.add_argument("--max-cartesian-acceleration", type=float, default=0.60)
    parser.add_argument("--lift-height", type=float, default=None)
    parser.add_argument("--target-contact-force", type=float, default=0.05)
    parser.add_argument(
        "--pose-completion-tolerance", type=float, default=0.003,
        help="Maximum final guide-position error for approach/lower/lift completion (m).",
    )
    parser.add_argument(
        "--pose-retries", type=int, default=2,
        help="Additional Cartesian approach/lower/lift attempts after an incomplete pose.",
    )
    parser.add_argument(
        "--auto-precision-control", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Scale XY entry and continuous-tracking tolerances to the physical pen "
            "diameter. This prevents a small pen from using millimetre-scale tolerances."
        ),
    )
    parser.add_argument(
        "--entry-xy-tolerance", type=float, default=None,
        help=(
            "Cross-track X accuracy requested before a vertical stroke starts (m); "
            "auto from pen diameter when omitted. This is not used to reject approach poses."
        ),
    )
    parser.add_argument(
        "--entry-along-track-tolerance", type=float, default=None,
        help="Along-line Y accuracy requested before drawing starts (m); auto when omitted.",
    )
    parser.add_argument(
        "--endpoint-xy-tolerance", type=float, default=None,
        help="Cross-track X accuracy requested at the final endpoint (m); auto when omitted.",
    )
    parser.add_argument(
        "--endpoint-along-track-tolerance", type=float, default=None,
        help="Along-line Y endpoint accuracy (m); auto when omitted.",
    )
    parser.add_argument(
        "--hard-pose-failure-tolerance", type=float, default=0.005,
        help=(
            "Only reject a rough approach/lift pose when the remaining guide error "
            "exceeds this value (m). Fine pen-width tolerances are applied later at the line."
        ),
    )
    parser.add_argument(
        "--cross-track-slowdown-error", type=float, default=None,
        help="Start slowing line progress at this lateral X error (m); auto when omitted.",
    )
    parser.add_argument(
        "--cross-track-stop-error", type=float, default=None,
        help="Pause line progress at this lateral X error (m); auto when omitted.",
    )
    parser.add_argument("--draw-start-settle-time", type=float, default=0.30)
    parser.add_argument("--draw-end-settle-time", type=float, default=0.20)
    parser.add_argument("--xy-stable-time", type=float, default=0.03)
    parser.add_argument(
        "--seat-correction-attempts", type=int, default=4,
        help="Maximum number of bounded measured seating corrections before drawing.",
    )
    parser.add_argument(
        "--maximum-seat-correction-depth", type=float, default=0.0020,
        help=(
            "Maximum one-shot deterministic extra lowering after measuring a positive "
            "tip-paper gap (m). This is not an open-ended contact search."
        ),
    )
    parser.add_argument("--seat-correction-margin", type=float, default=0.00010)
    parser.add_argument(
        "--overlay-tolerance", type=float, default=None,
        help="Distance tolerance used only for target_contact_tolerance_overlay.png (m).",
    )
    parser.add_argument("--overforce-limit", type=float, default=5.0)
    parser.add_argument("--overforce-relief-step", type=float, default=0.0010)
    parser.add_argument(
        "--overforce-policy", choices=("record", "skip-segment", "skip-stroke"),
        default="record",
    )

    # IK and arm controller.
    parser.add_argument("--ik-position-tolerance", type=float, default=0.0015)
    parser.add_argument("--ik-orientation-tolerance", type=float, default=0.020)
    parser.add_argument("--ik-iterations", type=int, default=120)
    parser.add_argument("--ik-damping", type=float, default=0.002)
    parser.add_argument("--ik-step-size", type=float, default=0.60)
    parser.add_argument("--orientation-weight", type=float, default=0.30)
    parser.add_argument(
        "--cartesian-position-gain", type=float, default=12.0,
        help="Legacy/common Cartesian position gain used when axis-specific gains are omitted.",
    )
    parser.add_argument(
        "--cross-track-position-gain", type=float, default=24.0,
        help="Cartesian gain for lateral X error of vertical lines.",
    )
    parser.add_argument(
        "--along-track-position-gain", type=float, default=12.0,
        help="Cartesian gain for motion along vertical-line Y.",
    )
    parser.add_argument(
        "--normal-position-gain", type=float, default=10.0,
        help="Cartesian gain for paper-normal Z motion.",
    )
    parser.add_argument("--cartesian-orientation-gain", type=float, default=4.0)
    parser.add_argument("--cartesian-damping", type=float, default=0.015)
    parser.add_argument("--maximum-joint-speed", type=float, default=1.2)
    parser.add_argument(
        "--force-filter-time-constant", type=float, default=0.03,
        help="Low-pass time constant for contact-force safety decisions (s).",
    )
    parser.add_argument(
        "--overforce-hold-time", type=float, default=0.04,
        help="Time filtered force must exceed the limit before overforce is declared (s).",
    )
    parser.add_argument(
        "--max-joint-position-lead", type=float, default=0.03,
        help="Maximum commanded joint-position lead over measured q (rad).",
    )
    parser.add_argument("--viewer-ink-radius", type=float, default=0.00035)
    parser.add_argument("--viewer-ink-min-spacing", type=float, default=0.00025)
    parser.add_argument("--viewer-ink-max-segments", type=int, default=8000)
    parser.add_argument(
        "--require-visible-ink", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Return a nonzero status when no physical ink is produced. Disabled by default "
            "so logs and simulated_drawing.png are always finalized for diagnosis."
        ),
    )
    parser.add_argument("--progress-width", type=int, default=18)
    parser.add_argument("--log-stride", type=int, default=10)
    parser.add_argument("--tracking-slowdown-error", type=float, default=0.0010,
                        help="Start slowing the vertical-line clock at this XY error (m).")
    parser.add_argument("--tracking-stop-error", type=float, default=0.0040,
                        help="Pause the vertical-line clock at this XY error (m).")
    parser.add_argument("--tracking-stall-timeout", type=float, default=8.0,
                        help="Abort one stroke after this much simulated time with no path progress (s).")
    parser.add_argument(
        "--continuous-stroke-timeout-factor", type=float, default=4.0,
        help="Hard per-stroke timeout multiplier relative to nominal draw time.",
    )
    return parser


# ---------------------------------------------------------------------------
# Include-aware MJCF patching: original base/table/paper stay unchanged.
# ---------------------------------------------------------------------------

def xml_parser() -> ET.XMLParser:
    try:
        return ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    except TypeError:
        return ET.XMLParser()


def parse_xml(path: Path) -> ET.ElementTree:
    return ET.parse(path, parser=xml_parser())


def format_floats(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def parse_floats(value: str | None) -> list[float]:
    if value is None or not value.strip():
        return []
    return [float(token) for token in value.split()]


def resolve_include(parent_path: Path, file_value: str) -> Path:
    path = Path(file_value).expanduser()
    if not path.is_absolute():
        path = parent_path.parent / path
    return path.resolve()


def discover_documents(top_path: Path) -> tuple[dict[Path, XmlDocument], list[Path]]:
    documents: dict[Path, XmlDocument] = {}
    order: list[Path] = []
    active: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in documents:
            return
        if path in active:
            raise ValueError(f"Cyclic MJCF include detected at {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Included MJCF file not found: {path}")
        active.add(path)
        tree = parse_xml(path)
        root = tree.getroot()
        if root.tag != "mujoco":
            raise ValueError(f"Not an MJCF document: {path}")
        includes: list[tuple[ET.Element, Path]] = []
        for element in root.iter("include"):
            file_value = element.get("file")
            if not file_value:
                raise ValueError(f"MJCF include without file attribute in {path}")
            child = resolve_include(path, file_value)
            includes.append((element, child))
        documents[path] = XmlDocument(path, tree, root, includes)
        order.append(path)
        for _, child in includes:
            visit(child)
        active.remove(path)

    visit(top_path.resolve())
    return documents, order


def clone_path_for(source: Path, top_input: Path, top_output: Path) -> Path:
    """Map every generated include into the runtime-model directory.

    The previous implementation wrote included Panda copies beside the source XML,
    which repopulated ``model/`` with generated files.  Keep the original include
    hierarchy, but place every generated document beside ``top_output`` instead.
    """
    if source == top_input:
        return top_output
    try:
        relative = source.resolve().relative_to(top_input.resolve().parent)
    except ValueError:
        relative = Path(source.name)
    generated_name = f"{relative.stem}{GENERATED_SUFFIX}{relative.suffix}"
    return top_output.parent / relative.parent / generated_name


def rewrite_compiler_resource_dirs(
    root: ET.Element, source_path: Path, output_path: Path
) -> None:
    """Retarget source asset directories from a generated runtime MJCF copy."""
    compiler = root.find("compiler")
    if compiler is None:
        return
    for attribute in ("assetdir", "meshdir", "texturedir"):
        raw = compiler.get(attribute)
        if not raw:
            continue
        resource = Path(raw).expanduser()
        if not resource.is_absolute():
            resource = (source_path.parent / resource).resolve()
        relative = os.path.relpath(resource, output_path.parent)
        compiler.set(attribute, Path(relative).as_posix())


def write_tree(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(path, encoding="utf-8", xml_declaration=True)


def find_named_elements(
    documents: dict[Path, XmlDocument], tag: str, name: str
) -> list[tuple[Path, ET.Element]]:
    matches: list[tuple[Path, ET.Element]] = []
    for path, document in documents.items():
        for element in document.root.iter(tag):
            if element.get("name") == name:
                matches.append((path, element))
    return matches


def find_site_document(documents: dict[Path, XmlDocument], site_name: str) -> Path:
    matches = find_named_elements(documents, "site", site_name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one site named {site_name!r}; found {len(matches)}: "
            + ", ".join(str(path) for path, _ in matches)
        )
    return matches[0][0]


def body_containing_direct_child(root: ET.Element, child: ET.Element) -> ET.Element:
    for body in root.iter("body"):
        if child in list(body):
            return body
    raise ValueError("Could not find the body directly containing the pen-tip site.")


def resize_round_geom(geom: ET.Element, radius: float) -> bool:
    """Change only a geom radial dimension while preserving length and pose."""
    values = parse_floats(geom.get("size"))
    if not values:
        return False
    geom_type = geom.get("type", "sphere")
    radius = max(1e-5, float(radius))
    if geom_type in {"sphere", "capsule", "cylinder"}:
        values[0] = radius
    elif geom_type == "box" and len(values) >= 2:
        values[0] = radius
        values[1] = radius
    else:
        return False
    geom.set("size", format_floats(values))
    return True


def reparent_original_pen_parts(
    hand: ET.Element,
    carriage: ET.Element,
    *,
    pen_tip_site_name: str,
    pen_contact_geom_name: str,
    pen_body_radius: float,
    pen_tip_radius: float,
) -> list[str]:
    """Move the existing rigid pen assembly under the passive spring carriage.

    The source scene's rigid pen geometry is preserved exactly.  Only its parent
    body changes, so the pen body, physical tip geom, and ``pen_tip`` site share
    one passive translational degree of freedom behind the guide block.
    """
    requested_names: list[str] = []
    for name in (
        "drawing_pen_body",
        pen_contact_geom_name,
        "drawing_pen_tip",
        pen_tip_site_name,
    ):
        if name and name not in requested_names:
            requested_names.append(name)

    found_by_name: dict[str, ET.Element] = {}
    for name in requested_names:
        matches = [
            element
            for element in hand.iter()
            if element.tag in {"body", "geom", "site"}
            and element.get("name") == name
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Expected at most one original pen element named {name!r}; "
                f"found {len(matches)}."
            )
        if matches:
            found_by_name[name] = matches[0]

    if pen_tip_site_name not in found_by_name:
        raise ValueError(
            f"The original rigid pen site {pen_tip_site_name!r} was not found "
            "under the Panda hand."
        )
    if pen_contact_geom_name not in found_by_name:
        raise ValueError(
            f"The original rigid pen contact geom {pen_contact_geom_name!r} "
            "was not found. The model patch will not replace it with a new tip."
        )

    selected = set(found_by_name.values())
    parent_map = {child: parent for parent in hand.iter() for child in parent}

    # Move only selected roots.  When drawing_pen_body is itself a body that
    # already contains the tip/site, moving that body automatically preserves
    # the complete hierarchy without moving descendants twice.
    selected_roots: list[ET.Element] = []
    for element in found_by_name.values():
        parent = parent_map.get(element)
        has_selected_ancestor = False
        while parent is not None and parent is not hand:
            if parent in selected:
                has_selected_ancestor = True
                break
            parent = parent_map.get(parent)
        if not has_selected_ancestor and element not in selected_roots:
            selected_roots.append(element)

    for element in selected_roots:
        parent = parent_map.get(element)
        if parent is None:
            raise RuntimeError(
                f"Could not find the parent of original pen element "
                f"{element.get('name')!r}."
            )
        parent.remove(element)
        carriage.append(element)

    # Preserve the original hierarchy, pose, length, and names, but make the
    # visible/contact pen realistically slim. Only radial dimensions change.
    body_element = found_by_name.get("drawing_pen_body")
    if body_element is not None:
        if body_element.tag == "geom":
            resize_round_geom(body_element, pen_body_radius)
        elif body_element.tag == "body":
            for geom in body_element.iter("geom"):
                if geom.get("name") != pen_contact_geom_name:
                    resize_round_geom(geom, pen_body_radius)

    contact_geom = found_by_name[pen_contact_geom_name]
    if contact_geom.tag != "geom":
        raise ValueError(
            f"Original pen contact element {pen_contact_geom_name!r} is not a geom."
        )
    resize_round_geom(contact_geom, pen_tip_radius)
    pen_site = found_by_name.get(pen_tip_site_name)
    if pen_site is not None:
        pen_site.set("size", f"{max(1e-5, float(pen_tip_radius)):.12g}")
    contact_geom.set("contype", "1")
    contact_geom.set("conaffinity", "1")
    contact_geom.set("condim", contact_geom.get("condim", "3"))
    contact_geom.set("priority", contact_geom.get("priority", "2"))
    contact_geom.set("margin", "0")
    contact_geom.set("gap", "0")
    contact_geom.set("friction", contact_geom.get("friction", "0.45 0.01 0.001"))

    return [name for name in requested_names if name in found_by_name]

def ensure_sensor(root: ET.Element, sensor_name: str, site_name: str) -> None:
    matches = [sensor for sensor in root.iter("touch") if sensor.get("name") == sensor_name]
    if len(matches) > 1:
        raise ValueError(f"Multiple touch sensors named {sensor_name!r} were found.")
    if matches:
        matches[0].set("site", site_name)
        return
    section = root.find("sensor")
    if section is None:
        section = ET.SubElement(root, "sensor")
    ET.SubElement(section, "touch", {"name": sensor_name, "site": site_name})


def add_passive_spring_pen(root: ET.Element, args: argparse.Namespace) -> tuple[str, list[str]]:
    sites = [site for site in root.iter("site") if site.get("name") == args.pen_tip_site_name]
    if len(sites) != 1:
        raise ValueError(
            f"Expected exactly one {args.pen_tip_site_name!r} site in the Panda XML; "
            f"found {len(sites)}."
        )
    old_site = sites[0]
    hand = body_containing_direct_child(root, old_site)
    hand_name = hand.get("name") or "<unnamed hand body>"

    ET.SubElement(
        hand,
        "geom",
        {
            "name": "spring_pen_guide_geom",
            "type": "box",
            "pos": format_floats((0.0, 0.0, args.guide_block_half_z)),
            "size": format_floats(
                (args.guide_block_half_x, args.guide_block_half_y, args.guide_block_half_z)
            ),
            "rgba": "0.20 0.45 0.75 1",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        hand,
        "site",
        {
            "name": args.guide_site_name,
            "type": "sphere",
            "pos": "0 0 0",
            "size": "0.003",
            "rgba": "0.1 0.7 1 0.25",
        },
    )
    carriage = ET.SubElement(
        hand,
        "body",
        {
            "name": args.spring_body_name,
            "pos": "0 0 0",
            # Isolate the specified spring law from the small pen weight.
            "gravcomp": "1",
        },
    )
    ET.SubElement(
        carriage,
        "inertial",
        {
            "mass": f"{args.pen_spring_mass:.12g}",
            "pos": "0 0 0.08",
            "diaginertia": "0.00006 0.00006 0.000004",
        },
    )
    ET.SubElement(
        carriage,
        "joint",
        {
            "name": args.spring_joint_name,
            "type": "slide",
            "axis": "0 0 1",
            "limited": "true",
            "range": format_floats((-args.pen_spring_travel, args.pen_spring_extension)),
            "springref": "0",
            "stiffness": f"{args.pen_spring_stiffness:.12g}",
            "damping": f"{args.pen_spring_damping:.12g}",
            "armature": f"{args.pen_spring_armature:.12g}",
        },
    )
    ET.SubElement(
        carriage,
        "geom",
        {
            "name": "spring_pen_spring_visual",
            "type": "cylinder",
            "pos": "0 0 0.020",
            "size": format_floats((min(args.pen_body_radius, 0.003), 0.015)),
            "rgba": "0.95 0.75 0.10 1",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )

    preserved = reparent_original_pen_parts(
        hand,
        carriage,
        pen_tip_site_name=args.pen_tip_site_name,
        pen_contact_geom_name=args.pen_contact_geom_name,
        pen_body_radius=args.pen_body_radius,
        pen_tip_radius=args.pen_tip_radius,
    )
    ensure_sensor(root, args.pen_touch_sensor_name, args.pen_tip_site_name)
    return hand_name, preserved

def find_paper_geom_id(model: Any, requested: str) -> int:
    if requested != "auto":
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, requested))
        if geom_id < 0:
            raise ValueError(f"Could not find paper geom {requested!r}.")
        return geom_id
    candidates: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and any(token in name.lower() for token in ("paper", "sheet", "canvas")):
            candidates.append(geom_id)
    if not candidates:
        raise ValueError("Could not auto-detect the paper collision geom.")
    return candidates[0]


def geom_top_z(model: Any, data: Any, geom_id: int) -> float:
    center = float(data.geom_xpos[geom_id, 2])
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        extent = 0.0
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        extent = float(sum(abs(rotation[2, j]) * size[j] for j in range(3)))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        extent = float(size[0])
    elif geom_type in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        vertical = abs(float(rotation[2, 2]))
        extent = vertical * float(size[1]) + math.sqrt(max(0.0, 1.0 - vertical**2)) * float(size[0])
    else:
        extent = float(model.geom_rbound[geom_id])
    return center + extent


def ensure_explicit_pen_paper_pair(root: ET.Element, pen_geom_name: str, paper_geom_name: str) -> None:
    section = root.find("contact")
    if section is None:
        section = ET.SubElement(root, "contact")
    for pair in section.findall("pair"):
        if {pair.get("geom1"), pair.get("geom2")} == {pen_geom_name, paper_geom_name}:
            pair.attrib.update({
                "condim": "3", "friction": "0.45 0.01 0.001 0.0001 0.0001",
                "margin": "0", "gap": "0", "solref": "0.005 1",
                "solimp": "0.95 0.99 0.001",
            })
            return
    ET.SubElement(
        section,
        "pair",
        {
            "name": "spring_pen_paper_pair",
            "geom1": pen_geom_name,
            "geom2": paper_geom_name,
            "condim": "3",
            "friction": "0.45 0.01 0.001 0.0001 0.0001",
            "margin": "0",
            "gap": "0",
            "solref": "0.005 1",
            "solimp": "0.95 0.99 0.001",
        },
    )


def compile_original_indices(model_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    finger_qpos: list[int] = []
    finger_dofs: list[int] = []
    for name in (args.finger_joint1_name, args.finger_joint2_name):
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if joint_id < 0:
            raise ValueError(f"Original model does not contain finger joint {name!r}.")
        finger_qpos.append(int(model.jnt_qposadr[joint_id]))
        finger_dofs.append(int(model.jnt_dofadr[joint_id]))
    gripper_id = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.gripper_actuator_name
    ))
    if gripper_id < 0:
        raise ValueError(f"Original model does not contain {args.gripper_actuator_name!r}.")
    return {
        "qpos_insert": max(finger_qpos) + 1,
        "dof_insert": max(finger_dofs) + 1,
        "finger_qpos": finger_qpos,
        "gripper_ctrl": gripper_id,
    }


def insert_token(tokens: list[float], index: int, value: float, label: str) -> list[float]:
    if not 0 <= index <= len(tokens):
        raise ValueError(f"Cannot insert {label} at {index}; vector has {len(tokens)} entries.")
    return tokens[:index] + [value] + tokens[index:]


def update_keyframes(
    roots: Iterable[ET.Element], indices: dict[str, Any], args: argparse.Namespace
) -> int:
    count = 0
    for root in roots:
        for keyframe in root.iter("keyframe"):
            for key in keyframe.findall("key"):
                qpos_raw = key.get("qpos")
                if qpos_raw is not None:
                    qpos = parse_floats(qpos_raw)
                    for index in indices["finger_qpos"]:
                        if int(index) >= len(qpos):
                            raise ValueError("Finger qpos index exceeds a keyframe qpos vector.")
                        qpos[int(index)] = args.gripper_finger_position
                    qpos = insert_token(qpos, int(indices["qpos_insert"]), 0.0, "spring qpos")
                    key.set("qpos", format_floats(qpos))
                qvel_raw = key.get("qvel")
                if qvel_raw is not None:
                    qvel = insert_token(
                        parse_floats(qvel_raw), int(indices["dof_insert"]), 0.0, "spring qvel"
                    )
                    key.set("qvel", format_floats(qvel))
                ctrl_raw = key.get("ctrl")
                if ctrl_raw is not None:
                    ctrl = parse_floats(ctrl_raw)
                    gripper_ctrl = int(indices["gripper_ctrl"])
                    if gripper_ctrl >= len(ctrl):
                        raise ValueError("Gripper control index exceeds keyframe ctrl vector.")
                    ctrl[gripper_ctrl] = args.gripper_control
                    key.set("ctrl", format_floats(ctrl))
                count += 1
    return count


def build_spring_pen_model(input_path: Path, output_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--spring-model must differ from --model.")
    original_model = mujoco.MjModel.from_xml_path(str(input_path))
    original_data = mujoco.MjData(original_model)
    reset_home(original_model, original_data)
    original_paper_id = find_paper_geom_id(original_model, args.paper_geom_name)
    paper_name = mujoco.mj_id2name(original_model, mujoco.mjtObj.mjOBJ_GEOM, original_paper_id)
    if not paper_name:
        raise ValueError("The selected paper collision geom has no name.")
    original_paper_top = geom_top_z(original_model, original_data, original_paper_id)

    documents, order = discover_documents(input_path)
    pen_document = find_site_document(documents, args.pen_tip_site_name)
    indices = compile_original_indices(input_path, args)
    clone_paths = {source: clone_path_for(source, input_path, output_path) for source in order}
    clone_trees: dict[Path, ET.ElementTree] = {}
    clone_roots: dict[Path, ET.Element] = {}
    for source in order:
        root_copy = copy.deepcopy(documents[source].root)
        rewrite_compiler_resource_dirs(root_copy, source, clone_paths[source])
        clone_roots[source] = root_copy
        clone_trees[source] = ET.ElementTree(root_copy)

    for source in order:
        copied_includes = list(clone_roots[source].iter("include"))
        source_includes = documents[source].includes
        if len(copied_includes) != len(source_includes):
            raise RuntimeError(f"Include-copy mismatch for {source}")
        parent_output = clone_paths[source]
        for copied, (_, child_source) in zip(copied_includes, source_includes):
            child_output = clone_paths[child_source]
            copied.set("file", Path(os.path.relpath(child_output, parent_output.parent)).as_posix())

    hand_name, removed = add_passive_spring_pen(clone_roots[pen_document], args)
    ensure_explicit_pen_paper_pair(
        clone_roots[input_path], args.pen_contact_geom_name, str(paper_name)
    )
    keyframes = update_keyframes(clone_roots.values(), indices, args)
    for source in reversed(order):
        write_tree(clone_trees[source], clone_paths[source])

    generated = mujoco.MjModel.from_xml_path(str(output_path))
    generated_data = mujoco.MjData(generated)
    reset_home(generated, generated_data)
    spring_id = int(mujoco.mj_name2id(generated, mujoco.mjtObj.mjOBJ_JOINT, args.spring_joint_name))
    if spring_id < 0:
        raise ValueError("Generated model is missing the passive spring joint.")
    if int(generated.jnt_type[spring_id]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
        raise ValueError("The passive pen joint is not prismatic.")
    generated_paper_id = find_paper_geom_id(generated, args.paper_geom_name)
    generated_paper_top = geom_top_z(generated, generated_data, generated_paper_id)
    if abs(generated_paper_top - original_paper_top) > 1e-9:
        raise ValueError("The spring model changed the original paper height.")
    for forbidden in ("panda_lift_joint", "gripper_pen_slide_joint"):
        if int(mujoco.mj_name2id(generated, mujoco.mjtObj.mjOBJ_JOINT, forbidden)) >= 0:
            raise ValueError(f"Forbidden old prismatic mechanism remains: {forbidden}")

    print(f"MODEL BUILD: {BUILD}", flush=True)
    print("Fixed Panda base and original desk/paper: preserved", flush=True)
    print(f"Gripper guide body: {hand_name}", flush=True)
    print(f"Preserved and spring-mounted original rigid pen parts: {removed}", flush=True)
    print(
        f"Pen dimensions: body diameter={2.0 * args.pen_body_radius:.4f} m, "
        f"tip diameter={2.0 * args.pen_tip_radius:.4f} m",
        flush=True,
    )
    print(
        f"Passive spring: joint={args.spring_joint_name}, stiffness={args.pen_spring_stiffness:.6g} N/m, "
        f"damping={args.pen_spring_damping:.6g} N s/m, "
        f"range=[{-args.pen_spring_travel:.6f}, {args.pen_spring_extension:.6f}] m",
        flush=True,
    )
    print(f"Paper geom={paper_name}, top z={generated_paper_top:.6f} m", flush=True)
    print(f"Generated model: {output_path}", flush=True)
    return {
        "paper_geom_name": str(paper_name),
        "paper_top_z_m": float(generated_paper_top),
        "keyframes_updated": int(keyframes),
        "source_model": str(input_path),
        "runtime_model_directory": str(output_path.parent),
        "generated_include_files": [str(clone_paths[source]) for source in order],
        "source_model_directory_left_unmodified": True,
    }


# ---------------------------------------------------------------------------
# Robot, path, logging, and image utilities.
# ---------------------------------------------------------------------------

def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def object_id(model: Any, object_type: Any, name: str, *, required: bool = True) -> int:
    identifier = int(mujoco.mj_name2id(model, object_type, name))
    if required and identifier < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return identifier


def reset_home(model: Any, data: Any) -> None:
    home_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    elif model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def rotation_aligning_vectors(
    source: np.ndarray, target: np.ndarray, fallback_axis: np.ndarray
) -> np.ndarray:
    """Return the minimum proper rotation that maps ``source`` onto ``target``."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= max(float(np.linalg.norm(source)), 1e-12)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    if cosine < -1.0 + 1e-10:
        axis = np.asarray(fallback_axis, dtype=float)
        axis = axis - source * float(axis @ source)
        if float(np.linalg.norm(axis)) < 1e-8:
            candidate = np.array([1.0, 0.0, 0.0])
            if abs(float(candidate @ source)) > 0.9:
                candidate = np.array([0.0, 1.0, 0.0])
            axis = candidate - source * float(candidate @ source)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    skew = np.array(
        [[0.0, -cross[2], cross[1]],
         [cross[2], 0.0, -cross[0]],
         [-cross[1], cross[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def downward_rotation_preserving_home_yaw(
    home_rotation: np.ndarray, spring_axis_local: np.ndarray
) -> np.ndarray:
    """Point the spring axis down using the smallest change from the home pose."""
    home_rotation = np.asarray(home_rotation, dtype=float).reshape(3, 3)
    spring_axis_local = np.asarray(spring_axis_local, dtype=float)
    spring_axis_local /= max(float(np.linalg.norm(spring_axis_local)), 1e-12)
    home_axis_world = home_rotation @ spring_axis_local
    alignment = rotation_aligning_vectors(
        home_axis_world,
        np.array([0.0, 0.0, -1.0]),
        home_rotation[:, 0],
    )
    target = alignment @ home_rotation
    u, _, vt = np.linalg.svd(target)
    target = u @ vt
    if np.linalg.det(target) < 0.0:
        u[:, -1] *= -1.0
        target = u @ vt
    return target


def rotation_error_vector(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = target @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ], dtype=float)
    if angle < 1e-8:
        return 0.5 * skew
    sine = math.sin(angle)
    if abs(sine) < 1e-8:
        diagonal = np.maximum((np.diag(relative) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) >= 1e-8 else np.array([1.0, 0.0, 0.0])
        return angle * axis
    return angle * skew / (2.0 * sine)


def actuator_position_limits(
    model: Any, joint_ids: np.ndarray, actuator_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(model.jnt_range[joint_ids], dtype=float).copy()
    lower = ranges[:, 0]
    upper = ranges[:, 1]
    for local, actuator_id in enumerate(actuator_ids):
        if bool(model.actuator_ctrllimited[int(actuator_id)]):
            lower[local] = max(lower[local], float(model.actuator_ctrlrange[actuator_id, 0]))
            upper[local] = min(upper[local], float(model.actuator_ctrlrange[actuator_id, 1]))
    return lower, upper


def solve_ik(
    model: Any, ik_data: Any, site_id: int, target_position: np.ndarray,
    target_rotation: np.ndarray, initial_q: np.ndarray, qpos_addresses: np.ndarray,
    dof_addresses: np.ndarray, lower_q: np.ndarray, upper_q: np.ndarray,
    args: argparse.Namespace,
) -> IKResult:
    q = np.asarray(initial_q, dtype=float).copy()
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    midpoint = 0.5 * (lower_q + upper_q)
    final_pe = math.inf
    final_oe = math.inf
    for iteration in range(1, int(args.ik_iterations) + 1):
        ik_data.qpos[qpos_addresses] = q
        ik_data.qvel[:] = 0.0
        mujoco.mj_forward(model, ik_data)
        position_error = target_position - ik_data.site_xpos[site_id]
        current_rotation = ik_data.site_xmat[site_id].reshape(3, 3)
        orientation_error = rotation_error_vector(target_rotation, current_rotation)
        final_pe = float(np.linalg.norm(position_error))
        final_oe = float(np.linalg.norm(orientation_error))
        if final_pe <= args.ik_position_tolerance and final_oe <= args.ik_orientation_tolerance:
            return IKResult(q.copy(), True, final_pe, final_oe, iteration)
        jacp.fill(0.0)
        jacr.fill(0.0)
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        jacobian = np.vstack((
            jacp[:, dof_addresses],
            args.orientation_weight * jacr[:, dof_addresses],
        ))
        error = np.concatenate((position_error, args.orientation_weight * orientation_error))
        regularized = jacobian @ jacobian.T + (args.ik_damping**2) * np.eye(6)
        delta = jacobian.T @ np.linalg.solve(regularized, error)
        pseudoinverse = jacobian.T @ np.linalg.solve(regularized, np.eye(6))
        nullspace = np.eye(7) - pseudoinverse @ jacobian
        delta += 0.01 * nullspace @ (midpoint - q)
        norm = float(np.linalg.norm(delta))
        if norm > 0.25:
            delta *= 0.25 / norm
        q = np.clip(q + args.ik_step_size * delta, lower_q + 1e-6, upper_q - 1e-6)
    return IKResult(q.copy(), False, final_pe, final_oe, int(args.ik_iterations))


def load_strokes(path: Path, max_strokes: int | None) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {
        "stroke_id", "point_index", "x_m", "y_m",
        "path_geometry", "is_graph_waypoint",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "The simulator requires the direct output of "
            "prepare_vertical_pen_paths_20260806.py. Missing columns: " + str(sorted(missing))
        )
    if frame.empty:
        raise ValueError("The prepared vertical-fill path CSV contains no points")
    geometry_values = set(frame["path_geometry"].dropna().astype(str).unique())
    accepted_geometry = {"dense_vertical_line_pen_width", "vertical_line_pen_width"}
    if not geometry_values or not geometry_values <= accepted_geometry:
        raise ValueError(
            "The trajectory source is not an accepted dense vertical-line path from "
            "prepare_vertical_pen_paths_20260806.py: path_geometry=" + str(sorted(geometry_values))
        )
    frame = frame.sort_values(["stroke_id", "point_index"], kind="stable")
    if max_strokes is not None:
        keep = sorted(frame["stroke_id"].unique())[:max(0, int(max_strokes))]
        frame = frame[frame["stroke_id"].isin(keep)]
    strokes: list[dict[str, Any]] = []
    for stroke_id, group in frame.groupby("stroke_id", sort=True):
        points = group[["x_m", "y_m"]].to_numpy(dtype=float)
        if not len(points):
            continue
        strokes.append({
            "stroke_id": int(stroke_id),
            "component_id": int(group["component_id"].iloc[0]) if "component_id" in group else -1,
            "points": points,
            "point_indices": group["point_index"].to_numpy(dtype=int),
            "is_graph_waypoint": (
                group["is_graph_waypoint"].to_numpy(dtype=int).astype(bool)
                if "is_graph_waypoint" in group
                else np.ones(len(group), dtype=bool)
            ),
            "graph_waypoint_node_ids": (
                group["graph_waypoint_node_ids"].fillna("").astype(str).to_numpy()
                if "graph_waypoint_node_ids" in group
                else np.asarray([""] * len(group), dtype=object)
            ),
            "is_closed": bool(int(group["is_closed"].iloc[0])) if "is_closed" in group else False,
        })
    if not strokes:
        raise ValueError("No strokes remained after filtering")
    return strokes


def quintic_blend(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def quintic_blend_derivative(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4


def target_speed(mode: str, args: argparse.Namespace) -> float:
    if mode == "draw":
        return args.draw_speed
    if mode in {"lower", "lift"}:
        return args.vertical_speed
    return args.transfer_speed


def append_rows(path: Path, rows: list[dict[str, Any]], *, new_file: bool) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(
        path, index=False, mode="w" if new_file else "a", header=new_file
    )
    rows.clear()


def project_xy(points: np.ndarray, configuration: dict[str, Any], width: int, height: int) -> np.ndarray:
    cx = float(configuration["paper_center_x_m"])
    cy = float(configuration["paper_center_y_m"])
    pw = float(configuration["paper_width_m"])
    ph = float(configuration["paper_height_m"])
    u = (points[:, 0] - (cx - pw / 2.0)) / pw * (width - 1)
    v = ((cy + ph / 2.0) - points[:, 1]) / ph * (height - 1)
    return np.column_stack((u, v)).round().astype(np.int32)


def draw_polyline_groups(
    canvas: np.ndarray, frame: pd.DataFrame, configuration: dict[str, Any],
    x_col: str, y_col: str, group_cols: list[str], color: tuple[int, int, int],
    thickness: int,
) -> None:
    if frame.empty:
        return
    h, w = canvas.shape[:2]
    for _, group in frame.groupby(group_cols, sort=True):
        points = group[[x_col, y_col]].to_numpy(dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]
        if not len(points):
            continue
        pixels = project_xy(points, configuration, w, h)
        if len(pixels) == 1:
            point_radius = max(1, int(round(0.5 * float(thickness))))
            cv2.circle(canvas, tuple(int(v) for v in pixels[0]), point_radius, color, -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [pixels], False, color, thickness, cv2.LINE_AA)


def _target_frame(strokes: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stroke in strokes:
        for local, point in enumerate(stroke["points"]):
            rows.append({
                "stroke_id": int(stroke["stroke_id"]),
                "point_index": int(stroke["point_indices"][local]),
                "x_m": float(point[0]),
                "y_m": float(point[1]),
                "is_graph_waypoint": int(bool(stroke.get("is_graph_waypoint", np.ones(len(stroke["points"]), dtype=bool))[local])),
            })
    return pd.DataFrame(rows)


def _draw_mask(
    shape: tuple[int, int], frame: pd.DataFrame, configuration: dict[str, Any],
    x_col: str, y_col: str, group_cols: list[str], thickness: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if frame.empty:
        return mask
    h, w = shape
    for _, group in frame.groupby(group_cols, sort=True):
        points = group[[x_col, y_col]].to_numpy(dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]
        if not len(points):
            continue
        pixels = project_xy(points, configuration, w, h)
        # Diagnostic masks must be binary and non-antialiased. The previous
        # LINE_AA + (>0) conversion promoted faint edge pixels to full cyan/red
        # errors and visually exaggerated small sub-pixel tracking offsets.
        if len(pixels) == 1:
            point_radius = max(1, int(round(0.5 * float(thickness))))
            cv2.circle(mask, tuple(int(v) for v in pixels[0]), point_radius, 255, -1, cv2.LINE_8)
        else:
            cv2.polylines(mask, [pixels], False, 255, thickness, cv2.LINE_8)
    return mask


def _render_thickness_px(paper_width_m: float, image_width_px: int, pen_tip_radius_m: float) -> int:
    paper_width_m = max(1e-9, float(paper_width_m))
    image_width_px = max(1, int(image_width_px))
    pen_tip_radius_m = max(1e-9, float(pen_tip_radius_m))
    return max(1, int(round((2.0 * pen_tip_radius_m / paper_width_m) * image_width_px)))


def _group_columns_for_contact(frame: pd.DataFrame) -> list[str]:
    return ["stroke_id", "ink_run_id"] if "ink_run_id" in frame.columns else ["stroke_id"]


def _load_graph_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        path = path.expanduser().resolve()
    except Exception:
        return {}
    if not path.is_file():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty or "parameter" not in frame.columns or "value" not in frame.columns:
        return {}
    summary: dict[str, Any] = {}
    for _, row in frame.iterrows():
        key = str(row["parameter"]).strip()
        value = row["value"]
        if not key:
            continue
        if isinstance(value, str):
            raw = value.strip()
            try:
                number = float(raw)
                value = int(number) if number.is_integer() else number
            except Exception:
                value = raw
        summary[key] = value
    return summary


def create_images(
    log: pd.DataFrame,
    strokes: list[dict[str, Any]],
    configuration: dict[str, Any],
    drawing_path: Path,
    trajectory_path: Path,
    overlay_path: Path,
    contact_ink_path: Path,
    contact_overlay_path: Path,
    contact_tolerance_overlay_path: Path,
    contact_centerline_overlay_path: Path,
    *,
    trajectory_thickness_px: int = 2,
    contact_thickness_px: int = 2,
    overlay_tolerance_m: float = 0.0,
) -> dict[str, Any]:
    """Write centerline diagnostics and physical pen-footprint drawing images.

    target_actual_overlay.png remains a thin centerline tracking diagnostic.
    target_contact_centerline_overlay.png compares contact-only centerlines.
    target_contact_overlay.png is a strict, non-antialiased physical-footprint
    comparison. target_contact_tolerance_overlay.png additionally ignores offsets
    smaller than overlay_tolerance_m. This separates true tracking excursions from
    rasterization fringes and expected sub-tolerance servo error.
    """
    width, height = 1400, 900
    white = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_rows = log[log["draw_command"] == 1].copy() if not log.empty else log.copy()
    if not draw_rows.empty and "time_s" in draw_rows.columns:
        draw_rows = draw_rows.sort_values("time_s")
    ink_column = "ink_active" if "ink_active" in draw_rows.columns else "contact_active"
    contact_rows = draw_rows[draw_rows[ink_column] == 1].copy() if not draw_rows.empty else draw_rows.copy()
    if not contact_rows.empty and "time_s" in contact_rows.columns:
        contact_rows = contact_rows.sort_values("time_s")
    target_frame = _target_frame(strokes)

    centerline_thickness = max(1, int(trajectory_thickness_px))
    physical_thickness = max(1, int(contact_thickness_px))

    trajectory = white.copy()
    if draw_rows.empty:
        draw_polyline_groups(
            trajectory, target_frame, configuration, "x_m", "y_m",
            ["stroke_id"], (180, 180, 180), centerline_thickness,
        )
    else:
        draw_polyline_groups(
            trajectory, draw_rows, configuration,
            "actual_x_m", "actual_y_m", ["stroke_id"], (0, 0, 0),
            centerline_thickness,
        )

    contact_ink = white.copy()
    if not contact_rows.empty:
        draw_polyline_groups(
            contact_ink, contact_rows, configuration,
            "actual_x_m", "actual_y_m", _group_columns_for_contact(contact_rows),
            (0, 0, 0), physical_thickness,
        )

    # Thin centerline tracking overlay: planned versus every executed draw sample.
    target_centerline_mask = _draw_mask(
        (height, width), target_frame, configuration, "x_m", "y_m", ["stroke_id"],
        centerline_thickness,
    )
    actual_centerline_mask = _draw_mask(
        (height, width), draw_rows, configuration, "actual_x_m", "actual_y_m", ["stroke_id"],
        centerline_thickness,
    )
    overlay = white.copy()
    target_only = (target_centerline_mask > 0) & (actual_centerline_mask == 0)
    actual_only = (actual_centerline_mask > 0) & (target_centerline_mask == 0)
    overlap = (target_centerline_mask > 0) & (actual_centerline_mask > 0)
    overlay[target_only] = np.array([0, 0, 255], dtype=np.uint8)
    overlay[actual_only] = np.array([255, 255, 0], dtype=np.uint8)
    overlay[overlap] = np.array([0, 0, 0], dtype=np.uint8)

    # Contact-only centerline diagnostic, kept separate from physical coverage.
    contact_centerline_mask = _draw_mask(
        (height, width), contact_rows, configuration, "actual_x_m", "actual_y_m",
        _group_columns_for_contact(contact_rows), centerline_thickness,
    )
    contact_centerline_overlay = white.copy()
    center_target_only = (target_centerline_mask > 0) & (contact_centerline_mask == 0)
    center_actual_only = (contact_centerline_mask > 0) & (target_centerline_mask == 0)
    center_overlap = (target_centerline_mask > 0) & (contact_centerline_mask > 0)
    contact_centerline_overlay[center_target_only] = np.array([0, 0, 255], dtype=np.uint8)
    contact_centerline_overlay[center_actual_only] = np.array([255, 255, 0], dtype=np.uint8)
    contact_centerline_overlay[center_overlap] = np.array([0, 0, 0], dtype=np.uint8)

    # Physical coverage overlay: target and actual are rendered with identical
    # pen-diameter thickness.  Red now means genuinely uncovered target footprint.
    target_physical_mask = _draw_mask(
        (height, width), target_frame, configuration, "x_m", "y_m", ["stroke_id"],
        physical_thickness,
    )
    contact_physical_mask = _draw_mask(
        (height, width), contact_rows, configuration, "actual_x_m", "actual_y_m",
        _group_columns_for_contact(contact_rows), physical_thickness,
    )
    contact_overlay = white.copy()
    target_only_contact = (target_physical_mask > 0) & (contact_physical_mask == 0)
    actual_only_contact = (contact_physical_mask > 0) & (target_physical_mask == 0)
    overlap_contact = (target_physical_mask > 0) & (contact_physical_mask > 0)
    contact_overlay[target_only_contact] = np.array([0, 0, 255], dtype=np.uint8)
    contact_overlay[actual_only_contact] = np.array([255, 255, 0], dtype=np.uint8)
    contact_overlay[overlap_contact] = np.array([0, 0, 0], dtype=np.uint8)

    # Tolerance-aware physical overlay. A pixel is cyan/red only when it lies
    # farther than the accepted Cartesian tracking tolerance from the opposite
    # footprint. The strict overlay above remains available for exact auditing.
    paper_width_m = max(1e-12, float(configuration["paper_width_m"]))
    tolerance_px = max(0, int(math.ceil(float(overlay_tolerance_m) / paper_width_m * width)))
    if tolerance_px > 0:
        kernel_size = 2 * tolerance_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        target_dilated = cv2.dilate(target_physical_mask, kernel) > 0
        actual_dilated = cv2.dilate(contact_physical_mask, kernel) > 0
    else:
        target_dilated = target_physical_mask > 0
        actual_dilated = contact_physical_mask > 0
    tolerant_target_only = (target_physical_mask > 0) & ~actual_dilated
    tolerant_actual_only = (contact_physical_mask > 0) & ~target_dilated
    tolerant_overlap = (target_physical_mask > 0) & (contact_physical_mask > 0)
    contact_tolerance_overlay = white.copy()
    contact_tolerance_overlay[tolerant_target_only] = np.array([0, 0, 255], dtype=np.uint8)
    contact_tolerance_overlay[tolerant_actual_only] = np.array([255, 255, 0], dtype=np.uint8)
    contact_tolerance_overlay[tolerant_overlap] = np.array([0, 0, 0], dtype=np.uint8)

    for path, image in (
        (drawing_path, contact_ink),
        (trajectory_path, trajectory),
        (contact_ink_path, contact_ink),
        (overlay_path, overlay),
        (contact_overlay_path, contact_overlay),
        (contact_tolerance_overlay_path, contact_tolerance_overlay),
        (contact_centerline_overlay_path, contact_centerline_overlay),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write image: {path}")

    physical_target_count = int(np.count_nonzero(target_physical_mask))
    physical_actual_count = int(np.count_nonzero(contact_physical_mask))
    physical_overlap_count = int(np.count_nonzero(overlap_contact))
    tolerant_target_only_count = int(np.count_nonzero(tolerant_target_only))
    tolerant_actual_only_count = int(np.count_nonzero(tolerant_actual_only))
    center_target_count = int(np.count_nonzero(target_centerline_mask))
    center_overlap_count = int(np.count_nonzero(center_overlap))
    return {
        "trajectory_draw_samples": int(len(draw_rows)),
        "contact_ink_samples": int(len(contact_rows)),
        "simulated_drawing_samples": int(len(contact_rows)),
        "output_trajectory_thickness_px": centerline_thickness,
        "output_contact_ink_thickness_px": physical_thickness,
        "physical_target_pixel_count": physical_target_count,
        "physical_contact_pixel_count": physical_actual_count,
        "physical_overlap_pixel_count": physical_overlap_count,
        "overlay_tolerance_m": float(overlay_tolerance_m),
        "overlay_tolerance_px": int(tolerance_px),
        "tolerance_target_only_pixel_count": tolerant_target_only_count,
        "tolerance_actual_only_pixel_count": tolerant_actual_only_count,
        "physical_target_coverage_ratio": float(physical_overlap_count / max(1, physical_target_count)),
        "physical_contact_precision_ratio": float(physical_overlap_count / max(1, physical_actual_count)),
        "contact_centerline_coverage_ratio": float(center_overlap_count / max(1, center_target_count)),
    }


def compute_metrics(log: pd.DataFrame, strokes: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare executed pen-tip XY trajectory with vertical-line graph waypoints.

    Node error uses the final actual sample associated with each graph waypoint.
    When a waypoint endpoint sample is unavailable, the nearest actual trajectory
    sample from the same stroke is used.  Edge error compares planned graph-edge
    chord lengths with chords between the matched actual node positions.  Total
    edge-length error compares the sums requested by the user.
    """
    draw = log[log["draw_command"] == 1].copy() if not log.empty else log.copy()
    draw = draw[np.isfinite(draw.get("actual_x_m", np.nan)) & np.isfinite(draw.get("actual_y_m", np.nan))] if not draw.empty else draw

    endpoint_lookup: dict[tuple[int, int], np.ndarray] = {}
    if not draw.empty:
        endpoint = draw.sort_values("time_s").groupby(
            ["stroke_id", "point_index"], sort=False, as_index=False
        ).tail(1)
        endpoint_lookup = {
            (int(row.stroke_id), int(row.point_index)): np.array([row.actual_x_m, row.actual_y_m], dtype=float)
            for row in endpoint.itertuples(index=False)
        }

    # The first node is normally reached during approach/contact search.  Use the
    # latest actual sample for that point even if contact was not stable.
    if not log.empty:
        first_rows = log[log["mode"].isin(["approach", "lower"])].copy()
        if not first_rows.empty:
            first_rows = first_rows.sort_values("time_s").groupby(
                ["stroke_id", "point_index"], sort=False, as_index=False
            ).tail(1)
            for row in first_rows.itertuples(index=False):
                endpoint_lookup.setdefault(
                    (int(row.stroke_id), int(row.point_index)),
                    np.array([row.actual_x_m, row.actual_y_m], dtype=float),
                )

    stroke_samples: dict[int, np.ndarray] = {}
    if not draw.empty:
        for sid, group in draw.sort_values("time_s").groupby("stroke_id", sort=False):
            xy = group[["actual_x_m", "actual_y_m"]].to_numpy(dtype=float)
            xy = xy[np.all(np.isfinite(xy), axis=1)]
            stroke_samples[int(sid)] = xy

    rows: list[dict[str, Any]] = []
    node_sq: list[float] = []
    node_abs: list[float] = []
    edge_sq: list[float] = []
    graph_total = 0.0
    endpoint_total = 0.0
    sampled_total = 0.0

    for sid, xy in stroke_samples.items():
        if len(xy) >= 2:
            sampled_total += float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())

    for stroke in strokes:
        sid = int(stroke["stroke_id"])
        points = np.asarray(stroke["points"], dtype=float)
        indices = np.asarray(stroke["point_indices"], dtype=int)
        graph_flags = np.asarray(
            stroke.get("is_graph_waypoint", np.ones(len(points), dtype=bool)), dtype=bool
        )
        graph_ids = np.asarray(
            stroke.get("graph_waypoint_node_ids", np.asarray([""] * len(points), dtype=object)),
            dtype=object,
        )
        samples = stroke_samples.get(sid, np.empty((0, 2), dtype=float))
        matched: list[np.ndarray | None] = []
        match_methods: list[str] = []
        for point, point_index in zip(points, indices):
            actual = endpoint_lookup.get((sid, int(point_index)))
            method = "waypoint_endpoint"
            if actual is None and len(samples):
                nearest = int(np.argmin(np.linalg.norm(samples - point[None, :], axis=1)))
                actual = samples[nearest].copy()
                method = "nearest_stroke_sample"
            matched.append(actual)
            match_methods.append(method if actual is not None else "missing")

        for local, (point, point_index, actual, method) in enumerate(
            zip(points, indices, matched, match_methods)
        ):
            is_graph_node = bool(graph_flags[local])
            node_error = float(np.linalg.norm(actual - point)) if actual is not None else math.nan
            node_squared = node_error * node_error if actual is not None else math.nan
            if actual is not None and is_graph_node:
                node_abs.append(node_error)
                node_sq.append(node_squared)

            planned_edge = actual_edge = edge_error = edge_squared = math.nan
            if local > 0:
                planned_edge = float(np.linalg.norm(points[local] - points[local - 1]))
                graph_total += planned_edge
                previous_actual = matched[local - 1]
                if actual is not None and previous_actual is not None:
                    actual_edge = float(np.linalg.norm(actual - previous_actual))
                    endpoint_total += actual_edge
                    edge_error = actual_edge - planned_edge
                    edge_squared = edge_error * edge_error
                    edge_sq.append(edge_squared)
            rows.append({
                "stroke_id": sid,
                "point_index": int(point_index),
                "match_method": method,
                "is_graph_node": int(is_graph_node),
                "graph_node_ids": str(graph_ids[local]),
                "included_in_node_metric": int(is_graph_node and actual is not None),
                "planned_x_m": float(point[0]),
                "planned_y_m": float(point[1]),
                "actual_x_m": float(actual[0]) if actual is not None else math.nan,
                "actual_y_m": float(actual[1]) if actual is not None else math.nan,
                "node_error_m": node_error,
                "node_squared_error_m2": node_squared,
                "planned_edge_length_m": planned_edge,
                "actual_edge_chord_length_m": actual_edge,
                "edge_length_error_m": edge_error,
                "edge_length_squared_error_m2": edge_squared,
            })

    node_mse = float(np.mean(node_sq)) if node_sq else None
    edge_mse = float(np.mean(edge_sq)) if edge_sq else None
    total_abs_error = abs(endpoint_total - graph_total) if graph_total > 0 else None
    total_rel_error = (100.0 * total_abs_error / graph_total) if total_abs_error is not None else None
    return pd.DataFrame(rows), {
        "metric_basis": "graph-set nodes versus executed pen-tip trajectory; graph-derived edge sums versus trajectory edge sums",
        "draw_samples": int(len(draw)),
        "graph_nodes_planned": int(sum(np.count_nonzero(np.asarray(s.get("is_graph_waypoint", np.ones(len(s["points"]), dtype=bool)), dtype=bool)) for s in strokes)),
        "evaluated_nodes": int(len(node_sq)),
        "evaluated_edges": int(len(edge_sq)),
        "node_position_mse_m2": node_mse,
        "node_position_rmse_mm": math.sqrt(node_mse) * 1000.0 if node_mse is not None else None,
        "node_position_mean_error_mm": float(np.mean(node_abs)) * 1000.0 if node_abs else None,
        "node_position_max_error_mm": float(np.max(node_abs)) * 1000.0 if node_abs else None,
        "edge_length_mse_m2": edge_mse,
        "edge_length_rmse_mm": math.sqrt(edge_mse) * 1000.0 if edge_mse is not None else None,
        "graph_total_edge_length_m": float(graph_total),
        "trajectory_endpoint_edge_sum_m": float(endpoint_total),
        "trajectory_sampled_path_length_m": float(sampled_total),
        "total_edge_length_absolute_error_m": float(total_abs_error) if total_abs_error is not None else None,
        "total_edge_length_relative_error_percent": float(total_rel_error) if total_rel_error is not None else None,
        "total_edge_length_ratio": endpoint_total / graph_total if graph_total > 0 else None,
    }

def main() -> None:
    args = build_parser().parse_args()
    if mujoco is None:
        raise ModuleNotFoundError("The mujoco Python package is required to run the simulator.")

    args.model = args.model.expanduser().resolve()
    args.scene_config = args.scene_config.expanduser().resolve()
    args.strokes = args.strokes.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.spring_model is None:
        args.spring_model = (
            args.output_dir / "runtime_model" /
            "drawing_scene_vertical_track_recovery_20260806.xml"
        ).resolve()
    else:
        requested_spring_model = args.spring_model.expanduser().resolve()
        # Protect the minimal source-model directory even when an old command still
        # supplies --spring-model model/drawing_scene_passive_....xml.
        if requested_spring_model.parent == args.model.parent:
            redirected = args.output_dir / "runtime_model" / requested_spring_model.name
            print(
                f"RUNTIME MODEL REDIRECT: {requested_spring_model} -> {redirected.resolve()}",
                flush=True,
            )
            args.spring_model = redirected.resolve()
        else:
            args.spring_model = requested_spring_model
    args.spring_model.parent.mkdir(parents=True, exist_ok=True)
    if args.graph_summary is not None:
        args.graph_summary = args.graph_summary.expanduser().resolve()
    for path, description in (
        (args.model, "Original MuJoCo model"),
        (args.scene_config, "Scene configuration"),
        (args.strokes, "drawing_strokes.csv"),
    ):
        require_file(path, description)
    for name in (
        "pen_spring_stiffness", "pen_spring_damping", "pen_spring_travel",
        "draw_speed", "transfer_speed", "vertical_speed", "minimum_segment_time",
        "overforce_limit", "ik_position_tolerance", "ik_orientation_tolerance",
        "target_contact_force",
        "pen_body_radius", "pen_tip_radius", "tracking_slowdown_error",
        "tracking_stop_error", "tracking_stall_timeout",
        "cartesian_position_gain", "cartesian_orientation_gain",
        "cartesian_damping", "maximum_joint_speed",
        "force_filter_time_constant", "overforce_hold_time", "max_joint_position_lead",
        "continuous_stroke_timeout_factor",
        "pose_completion_tolerance", "guide_press_depth",
        "lower_contact_gap_tolerance", "contact_settle_time",
        "draw_start_settle_time", "draw_end_settle_time", "xy_stable_time",
        "maximum_seat_correction_depth", "seat_correction_margin",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.tracking_stop_error <= args.tracking_slowdown_error:
        raise ValueError("--tracking-stop-error must exceed --tracking-slowdown-error")
    if args.log_stride <= 0:
        raise ValueError("--log-stride must be positive")
    if args.viewer_ink_radius <= 0 or args.viewer_ink_min_spacing <= 0:
        raise ValueError("Viewer ink radius and spacing must be positive")
    if args.viewer_ink_max_segments < 0:
        raise ValueError("--viewer-ink-max-segments cannot be negative")
    if args.progress_width < 10:
        raise ValueError("--progress-width must be at least 10")
    if args.pose_retries < 0:
        raise ValueError("--pose-retries cannot be negative")
    if args.seat_correction_attempts < 0:
        raise ValueError("--seat-correction-attempts cannot be negative")
    if args.viewer_startup_delay < 0:
        raise ValueError("--viewer-startup-delay cannot be negative")
    for optional_name in (
        "entry_xy_tolerance", "entry_along_track_tolerance",
        "endpoint_xy_tolerance", "endpoint_along_track_tolerance",
        "cross_track_slowdown_error", "cross_track_stop_error", "overlay_tolerance",
    ):
        value = getattr(args, optional_name)
        if value is not None and float(value) <= 0:
            raise ValueError(f"--{optional_name.replace('_', '-')} must be positive when supplied")

    total_start = time.perf_counter()
    if args.rebuild_spring_model or not args.spring_model.is_file():
        model_report = build_spring_pen_model(args.model, args.spring_model, args)
    else:
        model_report = {"reused": True, "generated_model": str(args.spring_model)}

    configuration = json.loads(args.scene_config.read_text(encoding="utf-8"))
    lift_height = (
        float(args.lift_height) if args.lift_height is not None
        else float(configuration["recommended_lift_height_m"])
    )
    strokes = load_strokes(args.strokes, args.max_strokes)

    log_path = args.output_dir / "simulation_log.csv"
    summary_path = args.output_dir / "simulation_summary.json"
    skipped_path = args.output_dir / "skipped_strokes_record.json"
    metrics_path = args.output_dir / "trajectory_graph_error_metrics.csv"
    drawing_path = args.output_dir / "simulated_drawing.png"
    trajectory_path = args.output_dir / "simulated_trajectory.png"
    contact_ink_path = args.output_dir / "simulated_contact_ink.png"
    overlay_path = args.output_dir / "target_actual_overlay.png"
    contact_overlay_path = args.output_dir / "target_contact_overlay.png"
    contact_tolerance_overlay_path = args.output_dir / "target_contact_tolerance_overlay.png"
    contact_centerline_overlay_path = args.output_dir / "target_contact_centerline_overlay.png"
    metrics_summary_path = args.output_dir / "trajectory_graph_error_summary.json"
    ik_path = args.output_dir / "ik_trajectory.npz"
    if log_path.exists():
        log_path.unlink()

    model = mujoco.MjModel.from_xml_path(str(args.spring_model))
    data = mujoco.MjData(model)
    reset_home(model, data)
    guide_site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, args.guide_site_name)
    pen_site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, args.pen_tip_site_name)
    pen_geom_id = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, args.pen_contact_geom_name)
    spring_joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, args.spring_joint_name)
    spring_qpos = int(model.jnt_qposadr[spring_joint_id])
    spring_dof = int(model.jnt_dofadr[spring_joint_id])
    touch_id = object_id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, args.pen_touch_sensor_name, required=False
    )
    touch_address = int(model.sensor_adr[touch_id]) if touch_id >= 0 else None
    paper_id = find_paper_geom_id(model, args.paper_geom_name)
    paper_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, paper_id) or "paper"
    paper_top_z = geom_top_z(model, data, paper_id)

    joint_ids = np.array([
        object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES
    ], dtype=int)
    actuator_ids = np.array([
        object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_ACTUATOR_NAMES
    ], dtype=int)
    qpos_addresses = model.jnt_qposadr[joint_ids]
    dof_addresses = model.jnt_dofadr[joint_ids]
    lower_q, upper_q = actuator_position_limits(model, joint_ids, actuator_ids)
    gripper_id = object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.gripper_actuator_name, required=False
    )
    if gripper_id >= 0:
        data.ctrl[gripper_id] = args.gripper_control

    # Measure the guide-to-tip offset at spring rest.  Align the spring axis with
    # world -Z through the smallest rotation from the home guide orientation; this
    # preserves the reachable home yaw instead of imposing an arbitrary 90-degree yaw.
    data.qpos[spring_qpos] = 0.0
    mujoco.mj_forward(model, data)
    offset_world = data.site_xpos[pen_site_id] - data.site_xpos[guide_site_id]
    guide_home_rotation = data.site_xmat[guide_site_id].reshape(3, 3).copy()
    offset_local = guide_home_rotation.T @ offset_world
    axis_local = np.asarray(model.jnt_axis[spring_joint_id], dtype=float)
    target_rotation = downward_rotation_preserving_home_yaw(
        guide_home_rotation, axis_local
    )
    # Use the target guide orientation to determine the world paper-normal
    # direction of the passive joint.  The reset/home pose need not already be
    # the drawing orientation.
    dz_dq = float((target_rotation @ axis_local)[2])
    if abs(dz_dq) < 0.5:
        raise RuntimeError(
            f"The passive pen joint is not aligned with paper normal: dz/dq={dz_dq:.6f}."
        )
    compression = min(
        args.target_contact_force / args.pen_spring_stiffness,
        0.80 * args.pen_spring_travel,
    )
    desired_spring_q = math.copysign(compression, dz_dq)
    physical_tip_radius = (
        float(model.geom_size[pen_geom_id, 0])
        if int(model.geom_type[pen_geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        else float(args.pen_tip_radius)
    )
    physical_pen_diameter = 2.0 * physical_tip_radius

    # Vertical lines have one meaningful lateral error: world X.  Do not use a
    # pen-scaled sub-millimetre tolerance as the completion gate for approach,
    # lower, or lift. Those long motions use pose_completion_tolerance. Fine
    # precision is applied only at the actual line start and during drawing.
    auto_entry_cross = max(1.0e-4, min(3.0e-4, 0.25 * physical_pen_diameter))
    auto_entry_along = max(5.0e-4, 1.00 * physical_pen_diameter)
    auto_cross_slow = max(1.5e-4, min(3.5e-4, 0.25 * physical_pen_diameter))
    auto_cross_stop = max(3.0 * auto_cross_slow, 0.75 * physical_pen_diameter)

    entry_xy_tolerance = (
        float(args.entry_xy_tolerance)
        if args.entry_xy_tolerance is not None
        else auto_entry_cross
    )
    entry_along_track_tolerance = (
        float(args.entry_along_track_tolerance)
        if args.entry_along_track_tolerance is not None
        else auto_entry_along
    )
    endpoint_xy_tolerance = (
        float(args.endpoint_xy_tolerance)
        if args.endpoint_xy_tolerance is not None
        else entry_xy_tolerance
    )
    endpoint_along_track_tolerance = (
        float(args.endpoint_along_track_tolerance)
        if args.endpoint_along_track_tolerance is not None
        else entry_along_track_tolerance
    )

    if args.auto_precision_control:
        effective_cross_track_slowdown_error = (
            float(args.cross_track_slowdown_error)
            if args.cross_track_slowdown_error is not None
            else auto_cross_slow
        )
        effective_cross_track_stop_error = (
            float(args.cross_track_stop_error)
            if args.cross_track_stop_error is not None
            else auto_cross_stop
        )
    else:
        effective_cross_track_slowdown_error = (
            float(args.cross_track_slowdown_error)
            if args.cross_track_slowdown_error is not None
            else float(args.tracking_slowdown_error)
        )
        effective_cross_track_stop_error = (
            float(args.cross_track_stop_error)
            if args.cross_track_stop_error is not None
            else float(args.tracking_stop_error)
        )

    effective_cross_track_stop_error = max(
        effective_cross_track_stop_error,
        1.5 * effective_cross_track_slowdown_error,
    )
    effective_along_track_slowdown_error = float(args.tracking_slowdown_error)
    effective_along_track_stop_error = max(
        float(args.tracking_stop_error),
        1.5 * effective_along_track_slowdown_error,
    )
    overlay_tolerance_m = (
        float(args.overlay_tolerance)
        if args.overlay_tolerance is not None
        else effective_cross_track_slowdown_error
    )
    draw_tip_z = paper_top_z + physical_tip_radius - args.pen_paper_penetration
    commanded_draw_tip_z = draw_tip_z - float(args.guide_press_depth)
    configured_draw_z = float(configuration.get("draw_target_tip_center_z_m", draw_tip_z))

    print(f"SIMULATOR BUILD: {BUILD}", flush=True)
    print("Fixed Panda base, original desk, and original paper: enabled", flush=True)
    print(
        "Control split: joints 1-7 move the rigid guide block through exact vertical-fill XY; "
        "the pen slides only through the passive spring joint inside the guide block.",
        flush=True,
    )
    print(
        f"Trajectory source: {args.strokes} | "
        f"vertical-fill path points={sum(len(stroke['points']) for stroke in strokes)} | "
        "path_geometry=dense_vertical_line_pen_width_track_recovery",
        flush=True,
    )
    print(
        f"Paper={paper_name}, top z={paper_top_z:.6f} m; physical contact tip-center z={draw_tip_z:.6f} m; "
        f"direct-press command z={commanded_draw_tip_z:.6f} m; scene-config draw z={configured_draw_z:.6f} m",
        flush=True,
    )
    if abs(configured_draw_z - draw_tip_z) > 5e-4:
        print(
            "SCENE CONFIG NOTE: draw_target_tip_center_z_m describes the source pen; "
            "the compiled runtime tip geometry is authoritative for this run.",
            flush=True,
        )
    print(
        f"Pen body diameter={2.0 * args.pen_body_radius:.4f} m; "
        f"physical tip diameter={2.0 * physical_tip_radius:.4f} m",
        flush=True,
    )
    print(
        f"Spring stiffness={args.pen_spring_stiffness:.6g} N/m, target force={args.target_contact_force:.6g} N, "
        f"target compression={compression:.6f} m, target q={desired_spring_q:.6f} m, dz/dq={dz_dq:.6f}",
        flush=True,
    )
    print(
        "Vertical-line tracking control: "
        f"approach/lift tolerance={float(args.pose_completion_tolerance):.6f} m; "
        f"entry cross/along={entry_xy_tolerance:.6f}/{entry_along_track_tolerance:.6f} m; "
        f"cross slowdown/stop={effective_cross_track_slowdown_error:.6f}/"
        f"{effective_cross_track_stop_error:.6f} m; "
        f"along slowdown/stop={effective_along_track_slowdown_error:.6f}/"
        f"{effective_along_track_stop_error:.6f} m",
        flush=True,
    )
    total_draw_length = float(sum(
        np.linalg.norm(np.diff(np.asarray(stroke["points"], dtype=float), axis=0), axis=1).sum()
        for stroke in strokes if len(stroke["points"]) > 1
    ))
    transfer_xy = 0.0
    previous_end = None
    for stroke in strokes:
        pts = np.asarray(stroke["points"], dtype=float)
        if len(pts) == 0:
            continue
        if previous_end is not None:
            transfer_xy += float(np.linalg.norm(pts[0] - previous_end))
        previous_end = pts[-1]
    nominal_draw_s = total_draw_length / max(args.draw_speed, 1e-9)
    nominal_transfer_s = transfer_xy / max(args.transfer_speed, 1e-9)
    print(
        f"Continuous stroke streaming: {sum(len(s['points']) for s in strokes)} prepared waypoints are interpolation knots, not stop targets; "
        f"draw length={total_draw_length:.3f} m, nominal draw time={format_duration(nominal_draw_s)}, "
        f"nominal XY transfer time={format_duration(nominal_transfer_s)}",
        flush=True,
    )

    ik_data = mujoco.MjData(model)
    viewer = None
    viewer_backend_used = "disabled"
    if args.viewer:
        print("Opening interactive MuJoCo viewer...", flush=True)
        try:
            if args.viewer_backend in {"x11", "wayland"}:
                # pyGLFW reads this before loading its bundled Linux library. X11
                # is more reliable than native Wayland for MuJoCo under WSLg.
                os.environ["PYGLFW_LIBRARY_VARIANT"] = args.viewer_backend
            if args.viewer_software_rendering:
                os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
            from mujoco import viewer as mujoco_viewer
            try:
                import glfw  # type: ignore
                if args.viewer_backend == "x11" and all(
                    hasattr(glfw, name) for name in ("init_hint", "PLATFORM", "PLATFORM_X11")
                ):
                    glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
                elif args.viewer_backend == "wayland" and all(
                    hasattr(glfw, name) for name in ("init_hint", "PLATFORM", "PLATFORM_WAYLAND")
                ):
                    glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_WAYLAND)
            except Exception:
                # Older GLFW bindings do not expose runtime platform selection;
                # PYGLFW_LIBRARY_VARIANT above remains the compatibility path.
                pass
            viewer = mujoco_viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False
            )
            startup_deadline = time.perf_counter() + max(0.0, args.viewer_startup_delay)
            while time.perf_counter() < startup_deadline:
                if not viewer.is_running():
                    raise RuntimeError("MuJoCo viewer closed during startup")
                viewer.sync()
                time.sleep(0.02)
            viewer_backend_used = args.viewer_backend
        except Exception as exc:
            display = os.environ.get("DISPLAY") or "<unset>"
            wayland = os.environ.get("WAYLAND_DISPLAY") or "<unset>"
            raise RuntimeError(
                "MuJoCo viewer launch failed. "
                f"backend={args.viewer_backend!r}, DISPLAY={display!r}, "
                f"WAYLAND_DISPLAY={wayland!r}. Original error: {type(exc).__name__}: {exc}"
            ) from exc
        print(
            f"MuJoCo viewer opened and synchronized (backend={viewer_backend_used}).",
            flush=True,
        )
    else:
        print("MuJoCo viewer disabled by --no-viewer.", flush=True)

    total_targets = sum(
        (len(stroke["points"]) + 2) if len(stroke["points"]) > 1 else 4
        for stroke in strokes
    )
    progress = SingleProgressBar(
        total_targets, enabled=not args.no_progress, width=args.progress_width
    )
    ink_renderer = ViewerInkRenderer(
        viewer,
        paper_z=paper_top_z,
        radius=args.viewer_ink_radius,
        minimum_spacing=args.viewer_ink_min_spacing,
        maximum_segments=args.viewer_ink_max_segments,
    )
    buffer: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    overforce_records: list[dict[str, Any]] = []
    ik_records: list[dict[str, Any]] = []
    runtime_notices: list[str] = []
    target_index = 0
    segment_id = 0
    log_new_file = True
    dt = float(model.opt.timestep)
    last_tip = data.site_xpos[pen_site_id].copy()
    log_step_counter = 0
    ink_run_id = 0
    ink_was_active = False

    filtered_contact_force_n = 0.0
    overforce_elapsed_s = 0.0

    def physical_contact_count() -> int:
        count = 0
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            if {int(contact.geom1), int(contact.geom2)} == {pen_geom_id, paper_id}:
                count += 1
        return count

    def instantaneous_forces() -> tuple[float, float, float]:
        sensor_force = (
            max(0.0, float(data.sensordata[touch_address]))
            if touch_address is not None else 0.0
        )
        spring_force = args.pen_spring_stiffness * abs(float(data.qpos[spring_qpos]))
        return sensor_force, spring_force, max(sensor_force, spring_force)

    def contact_state() -> tuple[bool, int, float, float, bool]:
        count = physical_contact_count()
        _, _, force = instantaneous_forces()
        sustained = overforce_elapsed_s >= args.overforce_hold_time
        return bool(count > 0), count, force, filtered_contact_force_n, sustained

    def update_contact_state() -> tuple[bool, int, float, float, bool]:
        nonlocal filtered_contact_force_n, overforce_elapsed_s
        count = physical_contact_count()
        _, _, force = instantaneous_forces()
        alpha = 1.0 - math.exp(-dt / max(args.force_filter_time_constant, dt))
        filtered_contact_force_n += alpha * (force - filtered_contact_force_n)
        if filtered_contact_force_n > args.overforce_limit:
            overforce_elapsed_s += dt
        else:
            overforce_elapsed_s = 0.0
        sustained = overforce_elapsed_s >= args.overforce_hold_time
        return bool(count > 0), count, force, filtered_contact_force_n, sustained

    def guide_target_for_tip(xy: np.ndarray, tip_z: float, spring_q: float) -> tuple[np.ndarray, np.ndarray]:
        tip = np.array([float(xy[0]), float(xy[1]), float(tip_z)], dtype=float)
        local = offset_local + axis_local * float(spring_q)
        guide = tip - target_rotation @ local
        return guide, tip

    def solve_target(target: PoseTarget) -> IKResult:
        ik_data.qpos[:] = data.qpos
        ik_data.qvel[:] = 0.0
        mujoco.mj_forward(model, ik_data)
        result = solve_ik(
            model, ik_data, guide_site_id, target.guide_position, target_rotation,
            data.qpos[qpos_addresses].copy(), qpos_addresses, dof_addresses,
            lower_q, upper_q, args,
        )
        ik_records.append({
            "target_index": target_index,
            "segment_id": target.segment_id,
            "stroke_id": target.stroke_id,
            "point_index": target.point_index,
            "mode": target.mode,
            "converged": result.converged,
            "position_error_m": result.position_error_m,
            "orientation_error_rad": result.orientation_error_rad,
            "iterations": result.iterations,
            "q": result.q.copy(),
        })
        return result

    def record_step(target: PoseTarget, q_ref: np.ndarray, phase: str, overforce_now: bool) -> None:
        nonlocal last_tip, log_new_file, log_step_counter, ink_run_id, ink_was_active
        log_step_counter += 1
        actual_tip = data.site_xpos[pen_site_id].copy()
        actual_guide = data.site_xpos[guide_site_id].copy()
        velocity = (actual_tip - last_tip) / max(dt, 1e-9)
        last_tip = actual_tip.copy()
        contact, count, force, filtered_force, sustained_overforce = contact_state()
        sensor_force, spring_force, _ = instantaneous_forces()
        spring_q = float(data.qpos[spring_qpos])
        compression_now = abs(spring_q)
        ink_active = bool(
            target.mode == "draw" and contact and not sustained_overforce and not overforce_now
        )
        if ink_active and not ink_was_active:
            ink_run_id += 1
        if ink_active:
            ink_renderer.add_sample(target.stroke_id, actual_tip)
        else:
            ink_renderer.break_stroke()
        ink_was_active = ink_active
        if (
            log_step_counter % args.log_stride != 0
            and not overforce_now
            and phase not in {"settle_final", "waypoint_endpoint"}
        ):
            return
        row: dict[str, Any] = {
            "time_s": float(data.time),
            "target_index": int(target_index),
            "segment_id": int(target.segment_id),
            "stroke_id": int(target.stroke_id),
            "component_id": int(target.component_id),
            "point_index": int(target.point_index),
            "mode": target.mode,
            "phase": phase,
            "draw_command": int(target.mode == "draw"),
            "contact_active": int(contact),
            "ink_active": int(ink_active),
            "ink_run_id": int(ink_run_id),
            "physical_contact_count": int(count),
            "touch_sensor_force_n": float(sensor_force),
            "raw_contact_force_n": float(force),
            "filtered_contact_force_n": float(filtered_force),
            "sustained_overforce": int(sustained_overforce),
            "spring_joint_position_m": spring_q,
            "spring_compression_m": compression_now,
            "spring_force_estimate_n": float(args.pen_spring_stiffness * compression_now),
            "desired_spring_position_m": float(target.spring_position),
            "desired_x_m": float(target.tip_position[0]),
            "desired_y_m": float(target.tip_position[1]),
            "desired_z_m": float(target.tip_position[2]),
            "actual_x_m": float(actual_tip[0]),
            "actual_y_m": float(actual_tip[1]),
            "actual_z_m": float(actual_tip[2]),
            "desired_guide_x_m": float(target.guide_position[0]),
            "desired_guide_y_m": float(target.guide_position[1]),
            "desired_guide_z_m": float(target.guide_position[2]),
            "actual_guide_x_m": float(actual_guide[0]),
            "actual_guide_y_m": float(actual_guide[1]),
            "actual_guide_z_m": float(actual_guide[2]),
            "position_error_m": float(np.linalg.norm(target.tip_position - actual_tip)),
            "xy_error_m": float(np.linalg.norm(target.tip_position[:2] - actual_tip[:2])),
            "cartesian_reference_error_m": float(
                np.linalg.norm(target.guide_position - actual_guide)
            ),
            "tip_speed_m_s": float(np.linalg.norm(velocity)),
            "overforce_event": int(overforce_now),
        }
        for i in range(7):
            row[f"q{i + 1}"] = float(data.qpos[qpos_addresses[i]])
            row[f"qref{i + 1}"] = float(q_ref[i])
        buffer.append(row)
        if len(buffer) >= 1000:
            append_rows(log_path, buffer, new_file=log_new_file)
            log_new_file = False

    commanded_q = data.qpos[qpos_addresses].copy()

    def servo_guide_step(
        reference_position: np.ndarray,
        feedforward_velocity: np.ndarray,
        posture_q: np.ndarray,
    ) -> tuple[np.ndarray, bool, int, float, float, bool]:
        """One resolved-rate Cartesian step using the earlier successful controller structure."""
        nonlocal commanded_q
        current_position = data.site_xpos[guide_site_id].copy()
        current_rotation = data.site_xmat[guide_site_id].reshape(3, 3).copy()
        position_error = reference_position - current_position
        orientation_error = rotation_error_vector(target_rotation, current_rotation)
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, data, jacp, jacr, guide_site_id)
        jacobian = np.vstack((
            jacp[:, dof_addresses],
            args.orientation_weight * jacr[:, dof_addresses],
        ))
        position_gains = np.array(
            [
                float(args.cross_track_position_gain),
                float(args.along_track_position_gain),
                float(args.normal_position_gain),
            ],
            dtype=float,
        )
        task_velocity = np.concatenate((
            feedforward_velocity + position_gains * position_error,
            args.orientation_weight * args.cartesian_orientation_gain * orientation_error,
        ))
        regularized = jacobian @ jacobian.T + (args.cartesian_damping ** 2) * np.eye(6)
        pseudoinverse = jacobian.T @ np.linalg.solve(regularized, np.eye(6))
        qdot = pseudoinverse @ task_velocity
        # The IK solution is only a redundant-joint posture preference; it never
        # defines the Cartesian path.
        null_projector = np.eye(7) - pseudoinverse @ jacobian
        qdot += 0.05 * null_projector @ (np.asarray(posture_q) - data.qpos[qpos_addresses])
        qdot = np.clip(qdot, -args.maximum_joint_speed, args.maximum_joint_speed)
        commanded_q = commanded_q + qdot * dt
        current_q = data.qpos[qpos_addresses].copy()
        commanded_q = np.clip(
            commanded_q,
            current_q - args.max_joint_position_lead,
            current_q + args.max_joint_position_lead,
        )
        commanded_q = np.clip(commanded_q, lower_q + 1e-6, upper_q - 1e-6)
        data.ctrl[actuator_ids] = commanded_q
        if gripper_id >= 0:
            data.ctrl[gripper_id] = args.gripper_control
        mujoco.mj_step(model, data)
        contact, count, raw, filtered, sustained = update_contact_state()
        return commanded_q.copy(), contact, count, raw, filtered, sustained

    def execute_pose(target: PoseTarget, q_goal: np.ndarray, *, break_on_overforce: bool) -> MotionResult:
        """Track approach/lower/lift targets in Cartesian space, not joint interpolation."""
        nonlocal log_new_file
        start_guide = data.site_xpos[guide_site_id].copy()
        delta = target.guide_position - start_guide
        distance = float(np.linalg.norm(delta))
        speed = min(max(target_speed(target.mode, args), 1e-6), args.max_cartesian_speed)
        speed_time = 1.875 * distance / speed
        accel_time = (
            math.sqrt((10.0 / math.sqrt(3.0)) * distance / max(args.max_cartesian_acceleration, 1e-6))
            if distance else 0.0
        )
        duration = max(args.minimum_segment_time, speed_time, accel_time)
        steps = max(2, int(math.ceil(duration / dt)))
        max_force = 0.0
        overforce_seen = False
        contact_seen = False
        q_ref = commanded_q.copy()
        for step in range(1, steps + 1):
            u = step / steps
            blend = quintic_blend(u)
            blend_rate = quintic_blend_derivative(u) / max(duration, dt)
            reference = start_guide + blend * delta
            feedforward = blend_rate * delta
            q_ref, contact, _, force, _, sustained = servo_guide_step(
                reference, feedforward, q_goal
            )
            contact_seen = contact_seen or contact
            max_force = max(max_force, force)
            overforce_seen = overforce_seen or sustained
            record_step(target, q_ref, "cartesian_track", sustained)
            if viewer is not None:
                if not viewer.is_running():
                    return MotionResult(False, overforce_seen, contact_seen, max_force, math.inf)
                viewer.sync()
            if sustained and break_on_overforce:
                break

        if not (overforce_seen and break_on_overforce):
            settle_steps = max(0, int(round(args.settle_time / dt)))
            for settle_index in range(settle_steps):
                q_ref, contact, _, force, _, sustained = servo_guide_step(
                    target.guide_position, np.zeros(3), q_goal
                )
                contact_seen = contact_seen or contact
                max_force = max(max_force, force)
                overforce_seen = overforce_seen or sustained
                record_step(
                    target, q_ref,
                    "settle_final" if settle_index + 1 == settle_steps else "settle",
                    sustained,
                )
                if viewer is not None:
                    if not viewer.is_running():
                        return MotionResult(False, overforce_seen, contact_seen, max_force, math.inf)
                    viewer.sync()
                if sustained and break_on_overforce:
                    break

        append_rows(log_path, buffer, new_file=log_new_file)
        log_new_file = False
        tip_error = float(np.linalg.norm(target.tip_position - data.site_xpos[pen_site_id]))
        guide_delta = target.guide_position - data.site_xpos[guide_site_id]
        guide_error = float(np.linalg.norm(guide_delta))
        guide_xy_error = float(np.linalg.norm(guide_delta[:2]))
        guide_z_error = abs(float(guide_delta[2]))
        pose_reached = (
            guide_xy_error <= float(args.pose_completion_tolerance)
            and guide_z_error <= float(args.pose_completion_tolerance)
        )
        return MotionResult(
            completed=pose_reached and not (overforce_seen and break_on_overforce),
            overforce=overforce_seen,
            contact=contact_seen,
            maximum_force_n=max_force,
            final_tip_error_m=tip_error,
        )

    def reach_pose_with_retries(
        target: PoseTarget,
        first_ik: IKResult,
        *,
        label: str,
        break_on_overforce: bool,
    ) -> tuple[MotionResult, np.ndarray]:
        """Execute a Cartesian pose and retry from the measured state when needed."""
        q_goal = first_ik.q.copy()
        result = execute_pose(target, q_goal, break_on_overforce=break_on_overforce)
        for retry_index in range(1, int(args.pose_retries) + 1):
            if result.completed or (result.overforce and break_on_overforce):
                break
            guide_error = float(np.linalg.norm(
                target.guide_position - data.site_xpos[guide_site_id]
            ))
            runtime_notices.append(
                f"POSE RETRY: {label}, attempt={retry_index}/{args.pose_retries}, "
                f"guide_error={guide_error:.6f} m"
            )
            retry_ik = solve_target(target)
            if not retry_ik.converged:
                runtime_notices.append(
                    f"POSE RETRY FAILED IK: {label}, "
                    f"position_error={retry_ik.position_error_m:.6f} m"
                )
                break
            q_goal = retry_ik.q.copy()
            result = execute_pose(target, q_goal, break_on_overforce=break_on_overforce)
        return result, q_goal


    def execute_continuous_stroke(
        stroke: dict[str, Any],
        points: np.ndarray,
        point_indices: np.ndarray,
        posture_q: np.ndarray,
        *,
        current_segment: int,
        stroke_ordinal: int,
        base_tip_z: float,
    ) -> MotionResult:
        """Stream one complete vertical-fill stroke continuously.

        All CSV waypoints remain ordered knots of the reference polyline, but
        they are not independent stop-and-settle targets.  Arc length advances
        at the commanded draw speed with acceleration limiting. The clock is
        slowed only by XY tracking error; contact loss never freezes the first
        stroke. This removes per-waypoint IK and settle overhead.
        """
        nonlocal target_index, log_new_file

        points = np.asarray(points, dtype=float)
        point_indices = np.asarray(point_indices, dtype=int)
        if len(points) <= 1:
            return MotionResult(True, False, False, 0.0, 0.0)

        # Keep every prepared waypoint, including zero-length duplicates, in the
        # progress and metric indexing.  Arc interpolation skips only degenerate
        # geometric intervals.
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
        total_length = float(cumulative[-1])
        if total_length <= 1e-12:
            q_ref = commanded_q.copy()
            for local in range(1, len(points)):
                target_index += 1
                guide, tip = guide_target_for_tip(points[local], base_tip_z, desired_spring_q)
                endpoint_target = PoseTarget(
                    guide, tip, desired_spring_q, int(stroke["stroke_id"]),
                    int(stroke["component_id"]), int(point_indices[local]),
                    "draw", current_segment,
                )
                record_step(endpoint_target, q_ref, "waypoint_endpoint", False)
                progress.update(
                    1,
                    f"stroke {stroke_ordinal}/{len(strokes)} waypoint {local + 1}/{len(points)}",
                )
            return MotionResult(True, False, contact_state()[0], 0.0, 0.0)

        # Retain the posture already reached at the first point. Solving IK for
        # the final point before the stroke biased the null-space toward the end
        # and could leave the physical pen near the first point.
        posture_reference = np.asarray(posture_q, dtype=float).copy()
        first_guide, first_tip = guide_target_for_tip(
            points[0], base_tip_z, desired_spring_q
        )
        current_target = PoseTarget(
            first_guide, first_tip, desired_spring_q, int(stroke["stroke_id"]),
            int(stroke["component_id"]), int(point_indices[0]), "draw",
            current_segment,
        )

        path_s = 0.0
        path_speed = 0.0
        last_crossed = 0
        tracking_stall_elapsed = 0.0
        elapsed = 0.0
        nominal_time = total_length / max(args.draw_speed, 1e-9)
        safety_time = max(
            nominal_time * args.continuous_stroke_timeout_factor,
            nominal_time + 2.0,
        )
        max_force = 0.0
        overforce_seen = False
        contact_seen = False
        q_ref = commanded_q.copy()

        # Anchor the physical tip at the exact first line endpoint before the
        # path clock starts. With a 0.6 mm pen, the previous 1-3 mm pose and
        # tracking tolerances allowed the reference to move while the tip was
        # still outside the planned footprint.
        stable_required = max(1, int(math.ceil(args.xy_stable_time / dt)))
        stable_count = 0
        start_steps = max(1, int(math.ceil(args.draw_start_settle_time / dt)))
        for _ in range(start_steps):
            q_ref, contact_now, _, force_now, _, overforce_now = servo_guide_step(
                current_target.guide_position, np.zeros(3), posture_reference
            )
            contact_seen = contact_seen or contact_now
            max_force = max(max_force, force_now)
            overforce_seen = overforce_seen or overforce_now
            record_step(current_target, q_ref, "draw_start_anchor", overforce_now)
            actual_start_xy = data.site_xpos[pen_site_id, :2]
            start_cross_error = abs(float(current_target.tip_position[0] - actual_start_xy[0]))
            start_along_error = abs(float(current_target.tip_position[1] - actual_start_xy[1]))
            start_aligned = (
                start_cross_error <= entry_xy_tolerance
                and start_along_error <= entry_along_track_tolerance
            )
            stable_count = stable_count + 1 if start_aligned else 0
            if viewer is not None:
                if not viewer.is_running():
                    return MotionResult(False, overforce_seen, contact_seen, max_force, math.inf)
                viewer.sync()
            if stable_count >= stable_required:
                break
            if overforce_now and args.overforce_policy != "record":
                break

        while path_s < total_length - 1e-12 and elapsed < safety_time:
            # Keep the guide at one fixed drawing height. The pen moves only
            # through the passive spring joint inside the guide block; there is
            # no separate contact-search or contact-reacquisition descent.
            # Spline progress is governed only by XY tracking quality.
            actual_tip_xy = data.site_xpos[pen_site_id, :2]
            cross_track_error = abs(float(current_target.tip_position[0] - actual_tip_xy[0]))
            along_track_error = abs(float(current_target.tip_position[1] - actual_tip_xy[1]))

            def error_scale(error: float, slowdown: float, stop: float) -> float:
                if error <= slowdown:
                    return 1.0
                if error >= stop:
                    return 0.0
                return (stop - error) / max(stop - slowdown, 1e-12)

            cross_scale = error_scale(
                cross_track_error,
                effective_cross_track_slowdown_error,
                effective_cross_track_stop_error,
            )
            along_scale = error_scale(
                along_track_error,
                effective_along_track_slowdown_error,
                effective_along_track_stop_error,
            )
            progress_scale = min(cross_scale, along_scale)

            if progress_scale <= 1e-6:
                tracking_stall_elapsed += dt
            else:
                tracking_stall_elapsed = 0.0
            if tracking_stall_elapsed >= args.tracking_stall_timeout:
                break

            remaining = max(0.0, total_length - path_s)
            braking_speed = math.sqrt(
                max(0.0, 2.0 * args.max_cartesian_acceleration * remaining)
            )
            requested_speed = progress_scale * min(
                args.draw_speed, args.max_cartesian_speed, braking_speed
            )
            if path_speed < requested_speed:
                path_speed = min(
                    requested_speed,
                    path_speed + args.max_cartesian_acceleration * dt,
                )
            else:
                path_speed = max(
                    requested_speed,
                    path_speed - args.max_cartesian_acceleration * dt,
                )
            path_s = min(total_length, path_s + path_speed * dt)

            # Locate the exact current segment of the prepared vertical-line polyline.
            upper = int(np.searchsorted(cumulative, path_s, side="right"))
            upper = min(max(1, upper), len(points) - 1)
            lower = upper - 1
            while upper < len(points) - 1 and cumulative[upper] <= cumulative[lower] + 1e-15:
                upper += 1
            interval = float(cumulative[upper] - cumulative[lower])
            alpha = 0.0 if interval <= 1e-15 else float(
                np.clip((path_s - cumulative[lower]) / interval, 0.0, 1.0)
            )
            xy = (1.0 - alpha) * points[lower] + alpha * points[upper]
            tangent = points[upper] - points[lower]
            tangent_norm = float(np.linalg.norm(tangent))
            tangent_xy = tangent / tangent_norm if tangent_norm > 1e-15 else np.zeros(2)

            guide, tip = guide_target_for_tip(xy, base_tip_z, desired_spring_q)
            current_target = PoseTarget(
                guide, tip, desired_spring_q, int(stroke["stroke_id"]),
                int(stroke["component_id"]), int(point_indices[upper]),
                "draw", current_segment,
            )
            feedforward = np.array(
                [tangent_xy[0] * path_speed, tangent_xy[1] * path_speed, 0.0],
                dtype=float,
            )
            q_ref, contact_now, _, force_now, _, overforce_now = servo_guide_step(
                guide, feedforward, posture_reference
            )
            elapsed += dt
            contact_seen = contact_seen or contact_now
            max_force = max(max_force, force_now)
            if overforce_now and not overforce_seen:
                overforce_records.append({
                    "stroke_id": int(stroke["stroke_id"]),
                    "point_index": int(point_indices[upper]),
                    "target_index": int(target_index),
                    "mode": "draw",
                    "raw_force_n": float(force_now),
                    "spring_q_m": float(data.qpos[spring_qpos]),
                    "simulation_time_s": float(data.time),
                })
            overforce_seen = overforce_seen or overforce_now
            record_step(current_target, q_ref, "continuous_cartesian_track", overforce_now)

            # Mark every vertical-fill waypoint crossed by the arc clock.
            crossed = int(np.searchsorted(cumulative, path_s + 1e-12, side="right") - 1)
            crossed = min(crossed, len(points) - 1)
            while last_crossed < crossed:
                last_crossed += 1
                target_index += 1
                endpoint_guide, endpoint_tip = guide_target_for_tip(
                    points[last_crossed], base_tip_z, desired_spring_q
                )
                endpoint_target = PoseTarget(
                    endpoint_guide, endpoint_tip, desired_spring_q,
                    int(stroke["stroke_id"]), int(stroke["component_id"]),
                    int(point_indices[last_crossed]), "draw", current_segment,
                )
                record_step(endpoint_target, q_ref, "waypoint_endpoint", overforce_now)
                progress.update(
                    1,
                    f"stroke {stroke_ordinal}/{len(strokes)} waypoint {last_crossed + 1}/{len(points)}",
                )

            if viewer is not None:
                if not viewer.is_running():
                    return MotionResult(False, overforce_seen, contact_seen, max_force, math.inf)
                viewer.sync()
            if overforce_now and args.overforce_policy != "record":
                break

        # Ensure the last endpoint is represented when numerical rounding leaves
        # the arc clock a fraction below the final cumulative length.
        completed = path_s >= total_length - 1e-9
        if completed:
            while last_crossed < len(points) - 1:
                last_crossed += 1
                target_index += 1
                endpoint_guide, endpoint_tip = guide_target_for_tip(
                    points[last_crossed], base_tip_z, desired_spring_q
                )
                endpoint_target = PoseTarget(
                    endpoint_guide, endpoint_tip, desired_spring_q,
                    int(stroke["stroke_id"]), int(stroke["component_id"]),
                    int(point_indices[last_crossed]), "draw", current_segment,
                )
                record_step(endpoint_target, q_ref, "waypoint_endpoint", overforce_seen)
                progress.update(
                    1,
                    f"stroke {stroke_ordinal}/{len(strokes)} waypoint {last_crossed + 1}/{len(points)}",
                )

        # Hold the exact final endpoint while still in draw mode. The old code
        # lifted as soon as the reference clock reached the endpoint, so the
        # physical tip could remain behind or swing outside the final capsule.
        if completed:
            endpoint_guide, endpoint_tip = guide_target_for_tip(
                points[-1], base_tip_z, desired_spring_q
            )
            endpoint_target = PoseTarget(
                endpoint_guide, endpoint_tip, desired_spring_q,
                int(stroke["stroke_id"]), int(stroke["component_id"]),
                int(point_indices[-1]), "draw", current_segment,
            )
            endpoint_stable = 0
            endpoint_steps = max(1, int(math.ceil(args.draw_end_settle_time / dt)))
            for _ in range(endpoint_steps):
                q_ref, contact_now, _, force_now, _, overforce_now = servo_guide_step(
                    endpoint_target.guide_position, np.zeros(3), posture_reference
                )
                contact_seen = contact_seen or contact_now
                max_force = max(max_force, force_now)
                overforce_seen = overforce_seen or overforce_now
                record_step(endpoint_target, q_ref, "draw_end_anchor", overforce_now)
                actual_endpoint_xy = data.site_xpos[pen_site_id, :2]
                endpoint_cross_error = abs(
                    float(endpoint_target.tip_position[0] - actual_endpoint_xy[0])
                )
                endpoint_along_error = abs(
                    float(endpoint_target.tip_position[1] - actual_endpoint_xy[1])
                )
                endpoint_aligned = (
                    endpoint_cross_error <= endpoint_xy_tolerance
                    and endpoint_along_error <= endpoint_along_track_tolerance
                )
                endpoint_stable = endpoint_stable + 1 if endpoint_aligned else 0
                if viewer is not None:
                    if not viewer.is_running():
                        return MotionResult(False, overforce_seen, contact_seen, max_force, math.inf)
                    viewer.sync()
                if endpoint_stable >= stable_required:
                    break
                if overforce_now and args.overforce_policy != "record":
                    break
            current_target = endpoint_target

        append_rows(log_path, buffer, new_file=log_new_file)
        log_new_file = False
        final_error = float(np.linalg.norm(current_target.tip_position - data.site_xpos[pen_site_id]))
        final_contact, _, _, _, _ = contact_state()
        return MotionResult(
            completed=completed and not (overforce_seen and args.overforce_policy != "record"),
            overforce=overforce_seen,
            contact=final_contact or contact_seen,
            maximum_force_n=max_force,
            final_tip_error_m=final_error,
        )

    def make_target(
        xy: np.ndarray, tip_z: float, spring_q: float, stroke: dict[str, Any],
        point_index: int, mode: str, current_segment: int, extra_guide_down: float = 0.0,
    ) -> PoseTarget:
        guide, tip = guide_target_for_tip(xy, tip_z, spring_q)
        guide = guide.copy()
        guide[2] -= max(0.0, extra_guide_down)
        return PoseTarget(
            guide, tip, spring_q, int(stroke["stroke_id"]), int(stroke["component_id"]),
            int(point_index), mode, int(current_segment),
        )

    interrupted = False
    try:
        for stroke_ordinal, stroke in enumerate(strokes, start=1):
            ink_renderer.break_stroke()
            points = np.asarray(stroke["points"], dtype=float)
            point_indices = np.asarray(stroke["point_indices"], dtype=int)
            sid = int(stroke["stroke_id"])
            skip_stroke = False

            # Approach above the first graph waypoint with the spring at rest.
            segment_id += 1
            target_index += 1
            approach = make_target(
                points[0], draw_tip_z + lift_height, 0.0, stroke,
                int(point_indices[0]), "approach", segment_id,
            )
            ik = solve_target(approach)
            if not ik.converged:
                skipped_records.append({
                    "stroke_id": sid, "failure_phase": "approach_ik",
                    "point_index": int(point_indices[0]), "reason": "Guide-block IK failed",
                    "position_error_m": ik.position_error_m,
                })
                skip_stroke = True
            else:
                approach_result, _ = reach_pose_with_retries(
                    approach,
                    ik,
                    label=f"stroke {sid} approach",
                    break_on_overforce=False,
                )
                if not approach_result.completed:
                    guide_error = float(np.linalg.norm(
                        approach.guide_position - data.site_xpos[guide_site_id]
                    ))
                    if guide_error > float(args.hard_pose_failure_tolerance):
                        skipped_records.append({
                            "stroke_id": sid,
                            "failure_phase": "approach_tracking",
                            "point_index": int(point_indices[0]),
                            "reason": (
                                "Guide remained outside the hard approach failure tolerance "
                                "after retries"
                            ),
                            "guide_position_error_m": guide_error,
                        })
                        runtime_notices.append(
                            f"APPROACH FAILED: stroke={sid}, guide_error={guide_error:.6f} m"
                        )
                        skip_stroke = True
                    else:
                        runtime_notices.append(
                            f"APPROACH RESIDUAL ACCEPTED: stroke={sid}, "
                            f"guide_error={guide_error:.6f} m; lower/start alignment will refine XY"
                        )
            progress.update(1, f"stroke {stroke_ordinal}/{len(strokes)} approach")

            # Lower the gripper-held guide block directly to the drawing pose.
            # The pen is not actuated: paper contact compresses the passive
            # spring joint while the guide reaches this fixed Cartesian height.
            segment_id += 1
            target_index += 1
            stroke_draw_tip_z = float(commanded_draw_tip_z)
            lower_target = make_target(
                points[0], stroke_draw_tip_z, desired_spring_q, stroke,
                int(point_indices[0]), "lower", segment_id,
            )
            if not skip_stroke:
                lower_ik = solve_target(lower_target)
                if not lower_ik.converged:
                    skipped_records.append({
                        "stroke_id": sid, "failure_phase": "lower_ik",
                        "point_index": int(point_indices[0]),
                        "reason": "Guide-block IK failed at the paper drawing pose",
                        "position_error_m": lower_ik.position_error_m,
                    })
                    skip_stroke = True
                else:
                    lower_result, lower_posture_q = reach_pose_with_retries(
                        lower_target,
                        lower_ik,
                        label=f"stroke {sid} lower",
                        break_on_overforce=True,
                    )

                    def settle_direct_press(target: PoseTarget, posture_q: np.ndarray, phase: str) -> None:
                        settle_steps = max(1, int(math.ceil(args.contact_settle_time / dt)))
                        for _ in range(settle_steps):
                            q_settle, _, _, _, _, sustained_settle = servo_guide_step(
                                target.guide_position, np.zeros(3), posture_q
                            )
                            record_step(target, q_settle, phase, sustained_settle)
                            if viewer is not None:
                                if not viewer.is_running():
                                    break
                                viewer.sync()
                            if sustained_settle:
                                break

                    settle_direct_press(lower_target, lower_posture_q, "direct_press_settle")
                    physical_now, count_now, raw_now, filtered_now, sustained_now = contact_state()
                    tip_bottom_gap = (
                        float(data.site_xpos[pen_site_id, 2])
                        - physical_tip_radius
                        - paper_top_z
                    )
                    seat_correction = 0.0
                    seat_attempts_used = 0

                    # Apply bounded measured corrections. Each step is based on
                    # the current geometric tip-paper gap, not on an open-ended
                    # blind search. The corrected guide height is retained for
                    # the complete vertical line.
                    while (
                        not physical_now
                        and not sustained_now
                        and not lower_result.overforce
                        and seat_attempts_used < int(args.seat_correction_attempts)
                        and seat_correction < float(args.maximum_seat_correction_depth) - 1e-12
                    ):
                        remaining_correction = (
                            float(args.maximum_seat_correction_depth) - seat_correction
                        )
                        requested_step = max(
                            float(args.seat_correction_margin),
                            max(0.0, float(tip_bottom_gap))
                            + float(args.seat_correction_margin),
                        )
                        correction_step = min(remaining_correction, requested_step)
                        if correction_step <= 1e-12:
                            break
                        seat_attempts_used += 1
                        seat_correction += correction_step
                        stroke_draw_tip_z = float(commanded_draw_tip_z) - seat_correction
                        corrected_target = make_target(
                            points[0], stroke_draw_tip_z, desired_spring_q, stroke,
                            int(point_indices[0]), "lower", segment_id,
                        )
                        corrected_ik = solve_target(corrected_target)
                        if not corrected_ik.converged:
                            runtime_notices.append(
                                f"SEAT CORRECTION IK FAILED: stroke={sid}, "
                                f"attempt={seat_attempts_used}, total={seat_correction:.6f} m"
                            )
                            break
                        corrected_result, corrected_posture_q = reach_pose_with_retries(
                            corrected_target, corrected_ik,
                            label=(
                                f"stroke {sid} measured seat correction "
                                f"{seat_attempts_used}/{args.seat_correction_attempts}"
                            ),
                            break_on_overforce=True,
                        )
                        lower_target = corrected_target
                        lower_result = corrected_result
                        lower_posture_q = corrected_posture_q
                        settle_direct_press(
                            lower_target, lower_posture_q, "measured_seat_settle"
                        )
                        physical_now, count_now, raw_now, filtered_now, sustained_now = contact_state()
                        tip_bottom_gap = (
                            float(data.site_xpos[pen_site_id, 2])
                            - physical_tip_radius
                            - paper_top_z
                        )
                        if physical_now:
                            break

                    contact_seated = bool(
                        physical_now
                        or tip_bottom_gap <= args.lower_contact_gap_tolerance
                    )
                    if sustained_now or lower_result.overforce or not contact_seated:
                        skipped_records.append({
                            "stroke_id": sid,
                            "failure_phase": "direct_lower_contact",
                            "point_index": int(point_indices[0]),
                            "reason": (
                                "The guide did not reach the fixed drawing pose with "
                                "safe physical pen-paper contact after bounded measured seating"
                            ),
                            "guide_position_error_m": float(np.linalg.norm(
                                lower_target.guide_position - data.site_xpos[guide_site_id]
                            )),
                            "tip_bottom_gap_m": float(tip_bottom_gap),
                            "seat_correction_m": float(seat_correction),
                            "seat_correction_attempts": int(seat_attempts_used),
                            "spring_q_m": float(data.qpos[spring_qpos]),
                            "physical_contact_count": int(count_now),
                            "raw_force_n": float(raw_now),
                            "filtered_force_n": float(filtered_now),
                        })
                        runtime_notices.append(
                            f"LOWER FAILED: stroke={sid}, tip_bottom_gap={tip_bottom_gap:.6f} m, "
                            f"seat_correction={seat_correction:.6f} m, attempts={seat_attempts_used}, "
                            f"physical_contacts={count_now}"
                        )
                        skip_stroke = True
                    else:
                        runtime_notices.append(
                            f"PAPER CONTACT: stroke={sid}, seat_correction={seat_correction:.6f} m, "
                            f"attempts={seat_attempts_used}, "
                            f"spring_q={float(data.qpos[spring_qpos]):.6f} m, force={raw_now:.6f} N"
                        )
            progress.update(1, f"stroke {stroke_ordinal}/{len(strokes)} lower")

            # Stream the entire vertical-fill stroke continuously.  All
            # waypoints remain ordered knots of the reference path, but they are
            # not independent stop-and-settle motions.
            if not skip_stroke:
                segment_id += 1
                result = execute_continuous_stroke(
                    stroke,
                    points,
                    point_indices,
                    commanded_q.copy(),
                    current_segment=segment_id,
                    stroke_ordinal=stroke_ordinal,
                    base_tip_z=stroke_draw_tip_z,
                )
                if result.overforce:
                    skipped_records.append({
                        "stroke_id": sid,
                        "failure_phase": "draw_overforce",
                        "point_index": int(point_indices[-1]),
                        "reason": f"Overforce policy {args.overforce_policy}",
                        "maximum_force_n": result.maximum_force_n,
                    })
                if not result.completed:
                    skipped_records.append({
                        "stroke_id": sid,
                        "failure_phase": "continuous_cartesian_tracking_timeout",
                        "point_index": int(point_indices[-1]),
                        "reason": "Continuous vertical-line stroke did not finish before the safety timeout or contact-loss limit",
                        "final_tip_error_m": result.final_tip_error_m,
                    })
            else:
                # Keep the one progress bar consistent even for a completely
                # unreachable stroke.
                progress.update(
                    max(0, len(points) - 1),
                    f"stroke {stroke_ordinal}/{len(strokes)} skipped",
                )

            # Lift the guide; the spring returns passively to q=0.
            segment_id += 1
            target_index += 1
            last_xy = points[-1]
            lift_target = make_target(
                last_xy, draw_tip_z + lift_height, 0.0, stroke,
                int(point_indices[-1]), "lift", segment_id,
            )
            lift_ik = solve_target(lift_target)
            if lift_ik.converged:
                execute_pose(lift_target, lift_ik.q, break_on_overforce=False)
            else:
                skipped_records.append({
                    "stroke_id": sid, "failure_phase": "lift_ik",
                    "point_index": int(point_indices[-1]),
                    "reason": "Lift IK failed; reset to home",
                })
                reset_home(model, data)
                last_tip = data.site_xpos[pen_site_id].copy()
            progress.update(1, f"stroke {stroke_ordinal}/{len(strokes)} lift")
    except KeyboardInterrupt:
        interrupted = True
        runtime_notices.append("Simulation interrupted; partial logs and images were finalized.")
    finally:
        append_rows(log_path, buffer, new_file=log_new_file)
        progress.finish("interrupted" if interrupted else "complete")
        if viewer is not None and not args.keep_viewer_open:
            viewer.close()

    log = pd.read_csv(log_path) if log_path.is_file() else pd.DataFrame(columns=LOG_COLUMNS)
    paper_width_for_render = float(configuration["paper_width_m"])
    trajectory_thickness_px = 2
    contact_thickness_px = _render_thickness_px(paper_width_for_render, 1400, float(args.pen_tip_radius))
    render_info = create_images(
        log,
        strokes,
        configuration,
        drawing_path,
        trajectory_path,
        overlay_path,
        contact_ink_path,
        contact_overlay_path,
        contact_tolerance_overlay_path,
        contact_centerline_overlay_path,
        trajectory_thickness_px=trajectory_thickness_px,
        contact_thickness_px=contact_thickness_px,
        overlay_tolerance_m=overlay_tolerance_m,
    )
    metrics_frame, metrics = compute_metrics(log, strokes)
    metrics_frame.to_csv(metrics_path, index=False)
    metrics_summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if ik_records:
        np.savez_compressed(
            ik_path,
            target_index=np.asarray([r["target_index"] for r in ik_records], dtype=int),
            segment_id=np.asarray([r["segment_id"] for r in ik_records], dtype=int),
            stroke_id=np.asarray([r["stroke_id"] for r in ik_records], dtype=int),
            point_index=np.asarray([r["point_index"] for r in ik_records], dtype=int),
            mode=np.asarray([r["mode"] for r in ik_records]),
            converged=np.asarray([r["converged"] for r in ik_records], dtype=bool),
            position_error_m=np.asarray([r["position_error_m"] for r in ik_records]),
            orientation_error_rad=np.asarray([r["orientation_error_rad"] for r in ik_records]),
            q=np.vstack([r["q"] for r in ik_records]),
        )

    skipped_payload = {
        "version": 1,
        "mechanism": "passive_prismatic_spring_pen",
        "records": skipped_records,
        "overforce_records": overforce_records,
    }
    skipped_path.write_text(json.dumps(skipped_payload, indent=2), encoding="utf-8")

    draw_rows = log[log["draw_command"] == 1] if not log.empty else log
    ink_column = "ink_active" if "ink_active" in draw_rows.columns else "contact_active"
    contact_rows = draw_rows[draw_rows[ink_column] == 1] if not draw_rows.empty else draw_rows
    ink_length_m = 0.0
    if len(contact_rows) >= 2:
        ink_length_groups = (
            ["stroke_id", "ink_run_id"] if "ink_run_id" in contact_rows.columns
            else ["stroke_id"]
        )
        for _, group in contact_rows.sort_values("time_s").groupby(
            ink_length_groups, sort=False
        ):
            xy = group[["actual_x_m", "actual_y_m"]].to_numpy(dtype=float)
            if len(xy) >= 2:
                ink_length_m += float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    visible_ink_ok = bool(len(contact_rows) >= 2 and ink_length_m >= 1e-5)
    if not visible_ink_ok:
        runtime_notices.append(
            "NO CONTACT INK: simulated_drawing.png was still written; "
            f"draw_rows={len(draw_rows)}, ink_rows={len(contact_rows)}, "
            f"ink_length_m={ink_length_m:.6e}"
        )
    graph_summary = _load_graph_summary(args.graph_summary)
    summary = {
        "build": BUILD,
        "status": (
            "interrupted" if interrupted else ("complete" if visible_ink_ok else "complete_no_contact_ink")
        ),
        "robot": "fixed_base_franka_emika_panda",
        "paper_setting": "original_scene_unchanged",
        "pen_mechanism": {
            "type": "original_rigid_pen_preserved_on_passive_prismatic_spring",
            "original_pen_parts_preserved": True,
            "stiffness_n_per_m": float(args.pen_spring_stiffness),
            "damping_n_s_per_m": float(args.pen_spring_damping),
            "compression_travel_m": float(args.pen_spring_travel),
            "extension_travel_m": float(args.pen_spring_extension),
            "pen_body_radius_m": float(args.pen_body_radius),
            "pen_tip_radius_m": float(args.pen_tip_radius),
            "target_contact_force_n": float(args.target_contact_force),
            "target_compression_m": float(compression),
            "maximum_measured_compression_m": (
                float(log["spring_compression_m"].max()) if len(log) else 0.0
            ),
            "maximum_estimated_spring_force_n": (
                float(log["spring_force_estimate_n"].max()) if len(log) else 0.0
            ),
        },
        "progress_bars": 1,
        "progress_display": "one live in-place terminal bar during MuJoCo",
        "trajectory_source": {
            "prepared_path_csv": str(args.strokes),
            "producer": "prepare_vertical_pen_paths_20260806.py",
            "path_geometry": "dense_vertical_line_pen_width",
            "waypoint_count": int(sum(len(stroke["points"]) for stroke in strokes)),
            "vertical_line_count": (
                int(graph_summary["vertical_line_count"])
                if graph_summary.get("vertical_line_count") is not None else None
            ),
            "pen_diameter_m": (
                float(graph_summary["pen_diameter_m"])
                if graph_summary.get("pen_diameter_m") is not None else 2.0 * float(args.pen_tip_radius)
            ),
            "line_overlap_fraction": (
                float(graph_summary["line_overlap_fraction"])
                if graph_summary.get("line_overlap_fraction") is not None else None
            ),
            "black_pixel_coverage_ratio": (
                float(graph_summary["black_pixel_coverage_ratio"])
                if graph_summary.get("black_pixel_coverage_ratio") is not None else None
            ),
            "graph_summary_csv": (str(args.graph_summary) if args.graph_summary is not None else None),
        },
        "paper_normal_control": {
            "mode": "direct_fixed_guide_height_with_passive_spring_compression",
            "contact_search_enabled": False,
            "physical_contact_tip_center_z_m": float(draw_tip_z),
            "commanded_direct_press_tip_center_z_m": float(commanded_draw_tip_z),
            "guide_press_depth_m": float(args.guide_press_depth),
            "maximum_seat_correction_depth_m": float(args.maximum_seat_correction_depth),
            "seat_correction_margin_m": float(args.seat_correction_margin),
            "target_spring_position_m": float(desired_spring_q),
            "pose_completion_tolerance_m": float(args.pose_completion_tolerance),
            "entry_cross_track_tolerance_m": float(entry_xy_tolerance),
            "entry_along_track_tolerance_m": float(entry_along_track_tolerance),
            "endpoint_cross_track_tolerance_m": float(endpoint_xy_tolerance),
            "endpoint_along_track_tolerance_m": float(endpoint_along_track_tolerance),
            "effective_cross_track_slowdown_error_m": float(effective_cross_track_slowdown_error),
            "effective_cross_track_stop_error_m": float(effective_cross_track_stop_error),
            "effective_along_track_slowdown_error_m": float(effective_along_track_slowdown_error),
            "effective_along_track_stop_error_m": float(effective_along_track_stop_error),
        },
        "trajectory_controller": "continuous_arc_length_streaming_of_exact_vertical_pen_width_lines_with_resolved_rate_cartesian_feedback",
        "prepared_waypoints_are_stop_targets": False,
        "prepared_waypoints_are_interpolation_knots": True,
        "nominal_draw_time_s": float(nominal_draw_s),
        "nominal_transfer_xy_time_s": float(nominal_transfer_s),
        "viewer_enabled": bool(args.viewer),
        "viewer_backend": viewer_backend_used,
        "viewer_software_rendering": bool(args.viewer_software_rendering),
        "live_viewer_ink": bool(args.viewer),
        "viewer_ink_segments": int(ink_renderer.segment_count),
        "rendering": {
            "simulated_trajectory_semantics": "all executed draw-command samples regardless of physical contact",
            "simulated_drawing_semantics": "physically contact-backed draw samples where ink_active==1",
            "simulated_contact_ink_semantics": "same pixels as simulated_drawing.png",
            "strict_overlay_mask_rasterization": "binary LINE_8 without antialias promotion",
            "tolerance_overlay_m": float(render_info["overlay_tolerance_m"]),
            "tolerance_overlay_px": int(render_info["overlay_tolerance_px"]),
            "tolerance_target_only_pixels": int(render_info["tolerance_target_only_pixel_count"]),
            "tolerance_actual_only_pixels": int(render_info["tolerance_actual_only_pixel_count"]),
            "output_trajectory_thickness_px": int(render_info["output_trajectory_thickness_px"]),
            "output_contact_ink_thickness_px": int(render_info["output_contact_ink_thickness_px"]),
            "trajectory_draw_samples": int(render_info["trajectory_draw_samples"]),
            "contact_ink_samples": int(render_info["contact_ink_samples"]),
            "simulated_drawing_samples": int(render_info["simulated_drawing_samples"]),
        },
        "contact_backed_ink_length_m": float(ink_length_m),
        "strokes_planned": len(strokes),
        "targets_completed": int(target_index),
        "simulation_log_rows": int(len(log)),
        "draw_samples": int(len(draw_rows)),
        "contact_draw_samples": int(len(contact_rows)),
        "contact_fraction_during_draw": (
            float(len(contact_rows) / len(draw_rows)) if len(draw_rows) else 0.0
        ),
        "maximum_raw_contact_force_n": (
            float(log["raw_contact_force_n"].max()) if len(log) else 0.0
        ),
        "skipped_record_count": len(skipped_records),
        "overforce_event_count": len(overforce_records),
        "visible_contact_ink_ok": bool(visible_ink_ok),
        "physical_target_coverage_ratio": float(render_info["physical_target_coverage_ratio"]),
        "physical_contact_precision_ratio": float(render_info["physical_contact_precision_ratio"]),
        "contact_centerline_coverage_ratio": float(render_info["contact_centerline_coverage_ratio"]),
        "runtime_notice_count": len(runtime_notices),
        "runtime_notices": runtime_notices,
        "trajectory_graph_error": metrics,
        "timing": {
            "total_wall_time_s": float(time.perf_counter() - total_start),
            "simulated_model_time_s": float(data.time),
        },
        "generated_model": model_report,
        "outputs": {
            "simulation_log": str(log_path),
            "simulation_summary": str(summary_path),
            "skipped_strokes_record": str(skipped_path),
            "trajectory_graph_error_metrics": str(metrics_path),
            "simulated_drawing": str(drawing_path),
            "simulated_trajectory": str(trajectory_path),
            "simulated_contact_ink": str(contact_ink_path),
            "target_actual_overlay": str(overlay_path),
            "target_contact_overlay": str(contact_overlay_path),
            "target_contact_tolerance_overlay": str(contact_tolerance_overlay_path),
            "target_contact_centerline_overlay": str(contact_centerline_overlay_path),
            "trajectory_graph_error_summary": str(metrics_summary_path),
            "ik_trajectory": str(ik_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Simulation outputs written: {drawing_path.name}, {trajectory_path.name}, "
        f"{summary_path.name}; contact_ink_ok={visible_ink_ok}; notices={len(runtime_notices)}",
        flush=True,
    )
    if runtime_notices:
        print(f"Motion diagnostics: {skipped_path}", flush=True)
    if viewer is not None and args.keep_viewer_open and viewer.is_running():
        print("Viewer remains open; close the MuJoCo window to exit.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
        viewer.close()
    if args.require_visible_ink and not visible_ink_ok:
        raise RuntimeError(
            "No visible contact-backed ink was produced; all diagnostic outputs were written."
        )


if __name__ == "__main__":
    main()
