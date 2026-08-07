#!/usr/bin/env python3
"""Prepare exact vertical pen-width graph edges as robot drawing strokes.

The graph maker already produces straight vertical world-coordinate edges. This
stage preserves those edges exactly, orders them by scan_order, and resamples
each line at the requested metric spacing. No curve fitting is applied because
spline fitting a two-point vertical edge would only reproduce the same line.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

BUILD = "vertical_pen_line_track_recovery_paths_20260806"
PATH_GEOMETRY = "dense_vertical_line_pen_width"


def parse_json_pair(value: Any, name: str) -> np.ndarray:
    try:
        points = np.asarray(json.loads(str(value)), dtype=float)
    except Exception as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc
    if points.shape != (2, 2):
        raise ValueError(f"{name} must contain exactly two 2D points")
    return points


def resample_segment(a: np.ndarray, b: np.ndarray, spacing_m: float) -> np.ndarray:
    length = float(np.linalg.norm(b - a))
    if length <= 1e-12:
        return np.vstack((a, b))
    intervals = max(1, int(math.ceil(length / float(spacing_m))))
    t = np.linspace(0.0, 1.0, intervals + 1)
    return a[None, :] + t[:, None] * (b - a)[None, :]


def save_preview(frame: pd.DataFrame, output_path: Path) -> None:
    width, height = 1400, 900
    margin = 40
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    min_x = float(frame["x_m"].min())
    max_x = float(frame["x_m"].max())
    min_y = float(frame["y_m"].min())
    max_y = float(frame["y_m"].max())
    scale = min(
        (width - 2 * margin) / max(max_x - min_x, 1e-12),
        (height - 2 * margin) / max(max_y - min_y, 1e-12),
    )

    def to_pixel(x: float, y: float) -> tuple[int, int]:
        u = int(round(margin + (x - min_x) * scale))
        v = int(round(height - margin - (y - min_y) * scale))
        return u, v

    for _, group in frame.groupby("stroke_id", sort=True):
        group = group.sort_values("point_index")
        points = np.array(
            [to_pixel(float(x), float(y)) for x, y in zip(group["x_m"], group["y_m"])],
            dtype=np.int32,
        )
        width_m = float(group["stroke_width_m"].iloc[0])
        thickness = max(1, int(round(width_m * scale)))
        if len(points) >= 2:
            cv2.polylines(canvas, [points], False, (0, 0, 0), thickness, cv2.LINE_AA)
            radius = max(1, thickness // 2)
            cv2.circle(canvas, tuple(points[0]), radius, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(points[-1]), radius, (0, 0, 0), -1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--graph-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.001)
    parser.add_argument("--vertical-tolerance-px", type=float, default=1e-6)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.spacing <= 0:
        parser.error("--spacing must be positive")
    if args.vertical_tolerance_px < 0:
        parser.error("--vertical-tolerance-px cannot be negative")

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    required_nodes = {"topo_node_id", "component_id", "line_id", "x_px", "y_px", "x_m", "y_m"}
    required_edges = {
        "topo_edge_id", "component_id", "line_id", "scan_order",
        "source_topo_node_id", "target_topo_node_id",
        "source_x_px", "source_y_px", "target_x_px", "target_y_px",
        "source_x_m", "source_y_m", "target_x_m", "target_y_m",
        "stroke_width_m", "direction",
    }
    missing_nodes = required_nodes - set(nodes.columns)
    missing_edges = required_edges - set(edges.columns)
    if missing_nodes:
        raise ValueError(f"Vertical-line node CSV is missing columns: {sorted(missing_nodes)}")
    if missing_edges:
        raise ValueError(f"Vertical-line edge CSV is missing columns: {sorted(missing_edges)}")
    if edges.empty:
        raise ValueError("Vertical-line edge CSV is empty")
    if nodes["topo_node_id"].duplicated().any():
        raise ValueError("Duplicate topo_node_id values were found")
    if edges["line_id"].duplicated().any():
        raise ValueError("Each vertical line must have exactly one edge row")

    edges = edges.sort_values(["scan_order", "line_id"], kind="stable")
    rows: list[dict[str, Any]] = []
    total_draw_length = 0.0
    pen_up_distance = 0.0
    previous_end: np.ndarray | None = None
    widths: list[float] = []

    for stroke_id, row in enumerate(edges.itertuples(index=False), start=1):
        source_px = np.array([float(row.source_x_px), float(row.source_y_px)], dtype=float)
        target_px = np.array([float(row.target_x_px), float(row.target_y_px)], dtype=float)
        if abs(source_px[0] - target_px[0]) > args.vertical_tolerance_px:
            raise ValueError(
                f"Line {int(row.line_id)} is not vertical: x0={source_px[0]}, x1={target_px[0]}"
            )
        source_m = np.array([float(row.source_x_m), float(row.source_y_m)], dtype=float)
        target_m = np.array([float(row.target_x_m), float(row.target_y_m)], dtype=float)
        world_points = resample_segment(source_m, target_m, args.spacing)
        # Use exactly the same interpolation parameter in pixel and world space.
        t = np.linspace(0.0, 1.0, len(world_points))
        pixel_points = source_px[None, :] + t[:, None] * (target_px - source_px)[None, :]

        if previous_end is not None:
            pen_up_distance += float(np.linalg.norm(world_points[0] - previous_end))
        previous_end = world_points[-1]
        draw_length = float(np.linalg.norm(target_m - source_m))
        total_draw_length += draw_length
        stroke_width = float(row.stroke_width_m)
        widths.append(stroke_width)
        source_component = int(getattr(row, "source_black_component_id", -1))
        for point_index, (pixel, world) in enumerate(zip(pixel_points, world_points)):
            endpoint = point_index == 0 or point_index == len(world_points) - 1
            node_id = (
                int(row.source_topo_node_id)
                if point_index == 0
                else int(row.target_topo_node_id)
                if point_index == len(world_points) - 1
                else None
            )
            rows.append(
                {
                    "stroke_id": stroke_id,
                    "component_id": int(row.component_id),
                    "line_id": int(row.line_id),
                    "scan_order": int(row.scan_order),
                    "grid_column_index": int(getattr(row, "grid_column_index", row.scan_order)),
                    "source_black_component_id": source_component,
                    "point_index": point_index,
                    "x_px": float(pixel[0]),
                    "y_px": float(pixel[1]),
                    "x_m": float(world[0]),
                    "y_m": float(world[1]),
                    "stroke_width_m": stroke_width,
                    "center_spacing_m": float(getattr(row, "center_spacing_m", stroke_width)),
                    "is_closed": 0,
                    "is_graph_waypoint": int(endpoint),
                    "graph_waypoint_node_ids": "" if node_id is None else str(node_id),
                    "path_geometry": PATH_GEOMETRY,
                    "incoming_segment_length_m": (
                        0.0
                        if point_index == 0
                        else float(np.linalg.norm(world_points[point_index] - world_points[point_index - 1]))
                    ),
                }
            )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    save_preview(output, args.preview)

    graph_summary: dict[str, Any] = {}
    if args.graph_summary is not None and args.graph_summary.is_file():
        try:
            graph_summary = json.loads(args.graph_summary.read_text(encoding="utf-8"))
        except Exception:
            graph_summary = {}
    summary = {
        "build": BUILD,
        "input_nodes": int(len(nodes)),
        "input_edges": int(len(edges)),
        "vertical_line_strokes": int(output["stroke_id"].nunique()),
        "trajectory_points": int(len(output)),
        "path_geometry": PATH_GEOMETRY,
        "line_geometry": "exact dense vertical straight segments on a globally phase-locked grid",
        "curve_spline_applied": False,
        "reason_no_curve_spline": "Every graph edge already contains exactly two collinear vertical endpoints.",
        "point_spacing_m": float(args.spacing),
        "pen_diameter_m": float(np.median(widths)),
        "minimum_pen_diameter_m": float(np.min(widths)),
        "maximum_pen_diameter_m": float(np.max(widths)),
        "estimated_draw_length_m": total_draw_length,
        "estimated_pen_up_xy_distance_m": pen_up_distance,
        "every_edge_drawn_once": True,
        "every_line_has_two_graph_endpoints": True,
        "source_graph_summary": graph_summary,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"PATH BUILD: {BUILD}")
        print(
            f"Vertical lines={summary['vertical_line_strokes']}, points={len(output)}, "
            f"draw length={total_draw_length:.3f} m, pen diameter={summary['pen_diameter_m']:.6f} m"
        )
        print(f"Wrote {args.output}")
        print(f"Wrote {args.preview}")
        print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
