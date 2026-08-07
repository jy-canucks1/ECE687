#!/usr/bin/env python3
"""Run black-pixel vertical-fill drawing with a fixed Panda and passive spring pen.

The runner starts vertical-line graph creation, path preparation, and the live MuJoCo viewer in one command. Graph creation and path
preparation write to log files quietly. The simulator owns one live in-place
progress bar; no checkpoint or resume files are created. The source model folder
contains only the original scene, Panda include, configuration, and assets; all
runtime-generated spring MJCF files are written under the selected output folder.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import pty
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

BUILD = "panda_vertical_track_recovery_workflow_20260806"


class WorkflowError(RuntimeError):
    pass


@dataclass
class StageResult:
    name: str
    command: list[str]
    log_file: str
    return_code: int
    elapsed_seconds: float
    expected_outputs: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: Path, root: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).expanduser().resolve()


def run_quiet_stage(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
    log_file: Path,
    expected_outputs: Sequence[Path],
    dry_run: bool,
    show_command: bool,
) -> StageResult:
    command_list = [str(item) for item in command]
    print(f"[{name}]", flush=True)
    if show_command:
        print(shlex.join(command_list), flush=True)
    started = time.perf_counter()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return_code = 0
    if not dry_run:
        with log_file.open("wb") as handle:
            process = subprocess.run(
                command_list,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return_code = int(process.returncode)
        if return_code != 0:
            raise WorkflowError(
                f"Stage {name!r} failed with exit code {return_code}. See {log_file}"
            )
        missing = [path for path in expected_outputs if not path.is_file()]
        if missing:
            raise WorkflowError(
                f"Stage {name!r} did not create: {', '.join(str(path) for path in missing)}"
            )
    return StageResult(
        name=name,
        command=command_list,
        log_file=str(log_file),
        return_code=return_code,
        elapsed_seconds=time.perf_counter() - started,
        expected_outputs=[str(path) for path in expected_outputs],
    )


def run_live_simulation(
    command: Sequence[str],
    *,
    cwd: Path,
    log_file: Path,
    expected_outputs: Sequence[Path],
    dry_run: bool,
    show_command: bool,
) -> StageResult:
    name = "mujoco_spring_pen_simulation"
    command_list = [str(item) for item in command]
    print(f"[{name}]", flush=True)
    if show_command:
        print(shlex.join(command_list), flush=True)
    started = time.perf_counter()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return_code = 0
    if not dry_run:
        with log_file.open("wb") as handle:
            master_fd, slave_fd = pty.openpty()
            child_env = os.environ.copy()
            if "--viewer" in command_list and "--viewer-backend" in command_list:
                backend_index = command_list.index("--viewer-backend") + 1
                if backend_index < len(command_list):
                    backend = command_list[backend_index]
                    if backend in {"x11", "wayland"}:
                        child_env["PYGLFW_LIBRARY_VARIANT"] = backend
            process = subprocess.Popen(
                command_list,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                env=child_env,
            )
            os.close(slave_fd)
            try:
                while True:
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            break
                        raise
                    if not chunk:
                        break
                    handle.write(chunk)
                    handle.flush()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            finally:
                try:
                    os.close(master_fd)
                except OSError:
                    pass
            return_code = int(process.wait())
        if return_code != 0:
            raise WorkflowError(
                f"Stage {name!r} failed with exit code {return_code}. See {log_file}"
            )
        missing = [path for path in expected_outputs if not path.is_file()]
        if missing:
            raise WorkflowError(
                f"Stage {name!r} did not create: {', '.join(str(path) for path in missing)}"
            )
    return StageResult(
        name=name,
        command=command_list,
        log_file=str(log_file),
        return_code=return_code,
        elapsed_seconds=time.perf_counter() - started,
        expected_outputs=[str(path) for path in expected_outputs],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_image", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("model/drawing_scene.xml"))
    parser.add_argument("--scene-config", type=Path, default=Path("model/drawing_scene_config.json"))

    # Black-pixel -> vertical pen-width graph.
    parser.add_argument("--black-threshold", type=int, default=160)
    parser.add_argument("--minimum-component-size", type=int, default=1)
    parser.add_argument("--black-mask-close-iterations", type=int, default=0)
    parser.add_argument("--line-overlap", type=float, default=0.50)
    parser.add_argument("--global-grid-phase", type=float, default=0.50)
    parser.add_argument("--coverage-repair-iterations", type=int, default=12)
    parser.add_argument("--minimum-centerline-length", type=float, default=0.00005)

    parser.add_argument("--paper-center-x", type=float, default=0.50)
    parser.add_argument("--paper-center-y", type=float, default=0.00)
    parser.add_argument("--paper-width", type=float, default=0.32)
    parser.add_argument("--paper-height", type=float, default=0.20)
    parser.add_argument("--paper-margin", type=float, default=0.01)
    parser.add_argument("--spacing", type=float, default=0.001)

    parser.add_argument("--pen-spring-stiffness", type=float, default=40.0)
    parser.add_argument("--pen-spring-damping", type=float, default=0.50)
    parser.add_argument("--pen-spring-travel", type=float, default=0.015)
    parser.add_argument("--pen-body-radius", type=float, default=0.0025)
    parser.add_argument("--pen-tip-radius", type=float, default=0.0010)
    parser.add_argument("--pen-paper-penetration", type=float, default=0.00020)
    parser.add_argument("--guide-press-depth", type=float, default=0.00150)
    parser.add_argument("--lower-contact-gap-tolerance", type=float, default=0.00025)
    parser.add_argument("--contact-settle-time", type=float, default=0.15)
    parser.add_argument("--target-contact-force", type=float, default=0.05)
    parser.add_argument("--overforce-limit", type=float, default=5.0)
    parser.add_argument("--pose-completion-tolerance", type=float, default=0.003)
    parser.add_argument("--pose-retries", type=int, default=2)
    parser.add_argument("--auto-precision-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--entry-xy-tolerance", type=float, default=None)
    parser.add_argument("--entry-along-track-tolerance", type=float, default=None)
    parser.add_argument("--endpoint-xy-tolerance", type=float, default=None)
    parser.add_argument("--endpoint-along-track-tolerance", type=float, default=None)
    parser.add_argument("--hard-pose-failure-tolerance", type=float, default=0.005)
    parser.add_argument("--cross-track-slowdown-error", type=float, default=None)
    parser.add_argument("--cross-track-stop-error", type=float, default=None)
    parser.add_argument("--draw-start-settle-time", type=float, default=0.30)
    parser.add_argument("--draw-end-settle-time", type=float, default=0.20)
    parser.add_argument("--xy-stable-time", type=float, default=0.03)
    parser.add_argument("--seat-correction-attempts", type=int, default=4)
    parser.add_argument("--maximum-seat-correction-depth", type=float, default=0.0025)
    parser.add_argument("--seat-correction-margin", type=float, default=0.00010)
    parser.add_argument("--overlay-tolerance", type=float, default=None)
    parser.add_argument("--cartesian-position-gain", type=float, default=12.0)
    parser.add_argument("--cross-track-position-gain", type=float, default=24.0)
    parser.add_argument("--along-track-position-gain", type=float, default=12.0)
    parser.add_argument("--normal-position-gain", type=float, default=10.0)
    parser.add_argument("--cartesian-damping", type=float, default=0.015)

    parser.add_argument("--draw-speed", type=float, default=0.005)
    parser.add_argument("--transfer-speed", type=float, default=0.030)
    parser.add_argument("--vertical-speed", type=float, default=0.005)
    parser.add_argument("--tracking-slowdown-error", type=float, default=0.0010)
    parser.add_argument("--tracking-stop-error", type=float, default=0.0040)
    parser.add_argument("--tracking-stall-timeout", type=float, default=8.0)
    parser.add_argument("--continuous-stroke-timeout-factor", type=float, default=4.0)
    parser.add_argument("--log-stride", type=int, default=10)
    parser.add_argument("--progress-width", type=int, default=18)

    parser.add_argument("--max-strokes", type=int, default=None)
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument("--viewer", dest="viewer", action="store_true")
    viewer_group.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.set_defaults(viewer=True)
    parser.add_argument("--keep-viewer-open", action="store_true")
    parser.add_argument("--viewer-startup-delay", type=float, default=1.50)
    parser.add_argument("--viewer-backend", choices=("auto", "x11", "wayland"), default="auto")
    parser.add_argument("--viewer-software-rendering", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-visible-ink", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--rebuild-spring-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-commands", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.project_root.expanduser().resolve()
    input_image = resolve(args.input_image, root)
    graph_script = resolve(Path("make_vertical_pen_graph_20260806.py"), root)
    path_script = resolve(Path("prepare_vertical_pen_paths_20260806.py"), root)
    simulator_script = resolve(Path("simulate_vertical_pen_track_recovery_20260806.py"), root)
    model = resolve(args.model, root)
    scene_config = resolve(args.scene_config, root)
    output_dir = resolve(args.output_dir, root)
    spring_model = (output_dir / "runtime_model" / "drawing_scene_vertical_track_recovery_20260806.xml").resolve()
    spring_model.parent.mkdir(parents=True, exist_ok=True)
    graph_dir = output_dir / "graph"
    logs_dir = output_dir / "logs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for required in (input_image, graph_script, path_script, simulator_script, model, scene_config):
        if not required.is_file():
            raise FileNotFoundError(required)
    source_panda = model.parent / "panda_drawing.xml"
    if not source_panda.is_file():
        raise FileNotFoundError(f"Required source Panda MJCF not found: {source_panda}")
    source_root = ET.parse(source_panda).getroot()
    named = {element.get("name") for element in source_root.iter() if element.get("name")}
    missing_names = sorted({"joint1", "pen_tip", "drawing_pen_tip"} - named)
    if missing_names:
        raise ValueError(
            f"{source_panda} is not the original Panda robot MJCF. Missing names: {missing_names}"
        )

    nodes_csv = graph_dir / "vertical_line_nodes.csv"
    edges_csv = graph_dir / "vertical_line_edges.csv"
    graph_summary_json = graph_dir / "vertical_fill_summary.json"
    graph_summary_csv = graph_dir / "vertical_fill_summary.csv"
    strokes_csv = output_dir / "drawing_strokes.csv"
    preview_path = output_dir / "drawing_strokes_preview.png"
    path_summary = output_dir / "path_summary.json"
    manifest_path = output_dir / "workflow_manifest.json"
    pen_diameter = 2.0 * float(args.pen_tip_radius)

    manifest: dict[str, object] = {
        "build": BUILD,
        "status": "running",
        "started_at_utc": utc_now(),
        "robot": "fixed_base_franka_emika_panda",
        "graph_design": "globally phase-locked dense pen-width vertical line components",
        "line_width_source": "2 * pen_tip_radius",
        "pen_diameter_m": pen_diameter,
        "paper": "original_desk_paper_unchanged",
        "runtime_spring_model": str(spring_model),
        "progress_bars": 1,
        "checkpointing": False,
        "stages": [],
    }
    stages: list[StageResult] = []

    try:
        graph_command = [
            sys.executable, str(graph_script), str(input_image),
            "--output-dir", str(graph_dir),
            "--black-threshold", str(args.black_threshold),
            "--minimum-component-size", str(args.minimum_component_size),
            "--close-iterations", str(args.black_mask_close_iterations),
            "--paper-center-x", str(args.paper_center_x),
            "--paper-center-y", str(args.paper_center_y),
            "--paper-width", str(args.paper_width),
            "--paper-height", str(args.paper_height),
            "--paper-margin", str(args.paper_margin),
            "--pen-diameter", str(pen_diameter),
            "--line-overlap", str(args.line_overlap),
            "--global-grid-phase", str(args.global_grid_phase),
            "--coverage-repair-iterations", str(args.coverage_repair_iterations),
            "--minimum-centerline-length", str(args.minimum_centerline_length),
            "--quiet",
        ]
        stages.append(run_quiet_stage(
            "black_pixels_to_vertical_graph", graph_command, cwd=root,
            log_file=logs_dir / "01_vertical_graph.log",
            expected_outputs=[
                nodes_csv, edges_csv, graph_summary_json, graph_summary_csv,
                graph_dir / "vertical_line_graph_preview.png",
                graph_dir / "vertical_line_coverage_preview.png",
            ],
            dry_run=args.dry_run, show_command=args.show_commands,
        ))

        path_command = [
            sys.executable, str(path_script),
            "--nodes", str(nodes_csv),
            "--edges", str(edges_csv),
            "--graph-summary", str(graph_summary_json),
            "--output", str(strokes_csv),
            "--preview", str(preview_path),
            "--summary", str(path_summary),
            "--spacing", str(args.spacing),
            "--quiet",
        ]
        stages.append(run_quiet_stage(
            "vertical_graph_to_paths", path_command, cwd=root,
            log_file=logs_dir / "02_vertical_paths.log",
            expected_outputs=[strokes_csv, preview_path, path_summary],
            dry_run=args.dry_run, show_command=args.show_commands,
        ))

        simulation_command = [
            sys.executable, str(simulator_script),
            "--model", str(model),
            "--spring-model", str(spring_model),
            "--scene-config", str(scene_config),
            "--strokes", str(strokes_csv),
            "--output-dir", str(output_dir),
            "--graph-summary", str(graph_summary_csv),
            "--pen-spring-stiffness", str(args.pen_spring_stiffness),
            "--pen-spring-damping", str(args.pen_spring_damping),
            "--pen-spring-travel", str(args.pen_spring_travel),
            "--pen-body-radius", str(args.pen_body_radius),
            "--pen-tip-radius", str(args.pen_tip_radius),
            "--pen-paper-penetration", str(args.pen_paper_penetration),
            "--guide-press-depth", str(args.guide_press_depth),
            "--lower-contact-gap-tolerance", str(args.lower_contact_gap_tolerance),
            "--contact-settle-time", str(args.contact_settle_time),
            "--target-contact-force", str(args.target_contact_force),
            "--overforce-limit", str(args.overforce_limit),
            "--pose-completion-tolerance", str(args.pose_completion_tolerance),
            "--pose-retries", str(args.pose_retries),
            "--auto-precision-control" if args.auto_precision_control else "--no-auto-precision-control",
            "--hard-pose-failure-tolerance", str(args.hard_pose_failure_tolerance),
            "--draw-start-settle-time", str(args.draw_start_settle_time),
            "--draw-end-settle-time", str(args.draw_end_settle_time),
            "--xy-stable-time", str(args.xy_stable_time),
            "--seat-correction-attempts", str(args.seat_correction_attempts),
            "--maximum-seat-correction-depth", str(args.maximum_seat_correction_depth),
            "--seat-correction-margin", str(args.seat_correction_margin),
            "--cartesian-position-gain", str(args.cartesian_position_gain),
            "--cross-track-position-gain", str(args.cross_track_position_gain),
            "--along-track-position-gain", str(args.along_track_position_gain),
            "--normal-position-gain", str(args.normal_position_gain),
            "--cartesian-damping", str(args.cartesian_damping),
            "--overforce-policy", "record",
            "--draw-speed", str(args.draw_speed),
            "--transfer-speed", str(args.transfer_speed),
            "--vertical-speed", str(args.vertical_speed),
            "--tracking-slowdown-error", str(args.tracking_slowdown_error),
            "--tracking-stop-error", str(args.tracking_stop_error),
            "--tracking-stall-timeout", str(args.tracking_stall_timeout),
            "--continuous-stroke-timeout-factor", str(args.continuous_stroke_timeout_factor),
            "--log-stride", str(args.log_stride),
            "--progress-width", str(args.progress_width),
            "--rebuild-spring-model" if args.rebuild_spring_model else "--no-rebuild-spring-model",
        ]
        if args.entry_xy_tolerance is not None:
            simulation_command.extend(["--entry-xy-tolerance", str(args.entry_xy_tolerance)])
        if args.entry_along_track_tolerance is not None:
            simulation_command.extend([
                "--entry-along-track-tolerance", str(args.entry_along_track_tolerance)
            ])
        if args.endpoint_xy_tolerance is not None:
            simulation_command.extend(["--endpoint-xy-tolerance", str(args.endpoint_xy_tolerance)])
        if args.endpoint_along_track_tolerance is not None:
            simulation_command.extend([
                "--endpoint-along-track-tolerance", str(args.endpoint_along_track_tolerance)
            ])
        if args.cross_track_slowdown_error is not None:
            simulation_command.extend([
                "--cross-track-slowdown-error", str(args.cross_track_slowdown_error)
            ])
        if args.cross_track_stop_error is not None:
            simulation_command.extend([
                "--cross-track-stop-error", str(args.cross_track_stop_error)
            ])
        if args.overlay_tolerance is not None:
            simulation_command.extend(["--overlay-tolerance", str(args.overlay_tolerance)])
        if args.max_strokes is not None:
            simulation_command.extend(["--max-strokes", str(args.max_strokes)])
        simulation_command.append("--viewer" if args.viewer else "--no-viewer")
        simulation_command.extend(["--viewer-startup-delay", str(args.viewer_startup_delay)])
        simulation_command.extend(["--viewer-backend", args.viewer_backend])
        simulation_command.append(
            "--viewer-software-rendering" if args.viewer_software_rendering
            else "--no-viewer-software-rendering"
        )
        simulation_command.append(
            "--require-visible-ink" if args.require_visible_ink
            else "--no-require-visible-ink"
        )
        if args.keep_viewer_open:
            simulation_command.append("--keep-viewer-open")
        if args.no_progress:
            simulation_command.append("--no-progress")

        stages.append(run_live_simulation(
            simulation_command, cwd=root,
            log_file=logs_dir / "03_mujoco_simulation.log",
            expected_outputs=[
                output_dir / "simulation_log.csv",
                output_dir / "simulation_summary.json",
                output_dir / "simulated_drawing.png",
                output_dir / "simulated_trajectory.png",
                output_dir / "simulated_contact_ink.png",
                output_dir / "target_actual_overlay.png",
                output_dir / "target_contact_overlay.png",
                output_dir / "target_contact_tolerance_overlay.png",
                output_dir / "target_contact_centerline_overlay.png",
                output_dir / "trajectory_graph_error_metrics.csv",
                output_dir / "trajectory_graph_error_summary.json",
            ],
            dry_run=args.dry_run, show_command=args.show_commands,
        ))
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at_utc"] = utc_now()
        manifest["stages"] = [asdict(stage) for stage in stages]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
