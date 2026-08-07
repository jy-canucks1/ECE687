#!/usr/bin/env python3
"""Convert every black image region into pen-width vertical graph lines.

Each output vertical segment is one independent graph component containing
exactly two nodes and one edge. Line centers are sampled on one globally
phase-locked dense raster. The default center spacing is one half of the physical
pen diameter, so one accidentally missed neighboring line does not open a full
pen-width gap. The rendered line width remains the physical pen diameter.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

BUILD = "vertical_pen_lines_track_recovery_20260806"


@dataclass(frozen=True)
class PaperMap:
    min_x_px: float
    max_x_px: float
    min_y_px: float
    max_y_px: float
    center_x_m: float
    center_y_m: float
    scale_m_per_px: float

    def pixel_to_world(self, x_px: float, y_px: float) -> tuple[float, float]:
        image_center_x = 0.5 * (self.min_x_px + self.max_x_px)
        image_center_y = 0.5 * (self.min_y_px + self.max_y_px)
        return (
            self.center_x_m + self.scale_m_per_px * (float(x_px) - image_center_x),
            self.center_y_m - self.scale_m_per_px * (float(y_px) - image_center_y),
        )


@dataclass(frozen=True)
class VerticalSegment:
    scan_x_px: float
    y_top_px: float
    y_bottom_px: float
    source_black_component_id: int
    scan_column_index: int


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_image_on_white(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image
    if image.shape[2] != 4:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    bgr = image[:, :, :3].astype(np.float32)
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    return np.clip(alpha * bgr + (1.0 - alpha) * white, 0, 255).astype(np.uint8)


def make_black_mask(
    image: np.ndarray,
    threshold: int,
    minimum_component_size: int,
    close_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray <= int(threshold)).astype(np.uint8)
    if close_iterations > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=int(close_iterations),
        )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    next_label = 1
    relabeled = np.zeros_like(labels, dtype=np.int32)
    kept_stats: list[np.ndarray] = [stats[0]]
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < int(minimum_component_size):
            continue
        component = labels == component_id
        cleaned[component] = 1
        relabeled[component] = next_label
        kept_stats.append(stats[component_id])
        next_label += 1
    return cleaned, relabeled, np.asarray(kept_stats, dtype=np.int32)


def calculate_paper_map(
    mask: np.ndarray,
    paper_center_x: float,
    paper_center_y: float,
    paper_width: float,
    paper_height: float,
    paper_margin: float,
) -> PaperMap:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("No black pixels remained after thresholding and filtering")
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    available_width = float(paper_width) - 2.0 * float(paper_margin)
    available_height = float(paper_height) - 2.0 * float(paper_margin)
    if available_width <= 0 or available_height <= 0:
        raise ValueError("Paper margin leaves no drawable paper area")
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min(available_width / span_x, available_height / span_y)
    return PaperMap(
        min_x_px=min_x,
        max_x_px=max_x,
        min_y_px=min_y,
        max_y_px=max_y,
        center_x_m=float(paper_center_x),
        center_y_m=float(paper_center_y),
        scale_m_per_px=float(scale),
    )


def true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    if values.size == 0 or not values.any():
        return []
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def phase_locked_component_centers(
    x_min: int,
    x_max: int,
    global_origin_px: float,
    spacing_px: float,
) -> list[tuple[float, int]]:
    """Return component columns aligned to one global compact raster.

    Restarting a raster at every connected component produces irregular horizontal
    phase shifts.  A single global origin keeps all vertical-line start points on
    the same dense grid.  Components narrower than one grid interval receive one
    center at their midpoint.
    """
    if spacing_px <= 0:
        raise ValueError("spacing_px must be positive")
    left = float(x_min)
    right = float(x_max)
    first_index = int(math.ceil((left - global_origin_px) / spacing_px - 1e-12))
    last_index = int(math.floor((right - global_origin_px) / spacing_px + 1e-12))
    centers: list[tuple[float, int]] = []
    if first_index <= last_index:
        for grid_index in range(first_index, last_index + 1):
            centers.append((global_origin_px + grid_index * spacing_px, grid_index))
    if not centers:
        midpoint = 0.5 * (left + right)
        grid_index = int(round((midpoint - global_origin_px) / spacing_px))
        centers = [(midpoint, grid_index)]
    return centers


def order_vertical_segments(raw: list[VerticalSegment]) -> list[VerticalSegment]:
    raw = list(raw)
    raw.sort(key=lambda s: (round(s.scan_x_px, 6), s.y_top_px, s.y_bottom_px, s.source_black_component_id))
    grouped: dict[float, list[VerticalSegment]] = {}
    for segment in raw:
        grouped.setdefault(round(segment.scan_x_px, 6), []).append(segment)
    ordered: list[VerticalSegment] = []
    for column_number, x_key in enumerate(sorted(grouped)):
        column_segments = grouped[x_key]
        column_segments.sort(key=lambda s: (s.y_top_px, s.y_bottom_px))
        if column_number % 2:
            column_segments.reverse()
        ordered.extend(column_segments)
    return ordered


def build_vertical_segments(
    mask: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    pen_diameter_px: float,
    overlap_fraction: float,
    minimum_centerline_length_px: float,
    global_grid_phase: float,
) -> list[VerticalSegment]:
    radius_px = 0.5 * float(pen_diameter_px)
    spacing_px = max(0.25, float(pen_diameter_px) * (1.0 - float(overlap_fraction)))
    phase = float(np.clip(global_grid_phase, 0.0, 1.0))
    nonzero_x = np.flatnonzero(np.any(mask > 0, axis=0))
    if nonzero_x.size == 0:
        return []
    global_left = float(nonzero_x.min())
    global_origin = global_left + phase * spacing_px
    raw: list[VerticalSegment] = []

    for source_component_id in range(1, len(stats)):
        component = labels == source_component_id
        if not component.any():
            continue
        x_min = int(stats[source_component_id, cv2.CC_STAT_LEFT])
        x_max = x_min + int(stats[source_component_id, cv2.CC_STAT_WIDTH]) - 1
        centers = phase_locked_component_centers(
            x_min, x_max, global_origin, spacing_px
        )
        for local_column_index, (center_x, grid_column_index) in enumerate(centers):
            strip_x0 = max(0, int(math.floor(center_x - radius_px)))
            strip_x1 = min(mask.shape[1] - 1, int(math.ceil(center_x + radius_px)))
            occupancy = np.any(component[:, strip_x0 : strip_x1 + 1], axis=1)
            for run_start, run_end in true_runs(occupancy):
                # A round pen cap covers approximately one radius beyond each
                # centerline endpoint.  Keep the centerline inside the black run.
                top = float(run_start) + radius_px
                bottom = float(run_end) - radius_px
                if bottom <= top:
                    center_y = 0.5 * (float(run_start) + float(run_end))
                    half = 0.5 * max(float(minimum_centerline_length_px), 0.1)
                    top = center_y - half
                    bottom = center_y + half
                top = float(np.clip(top, 0.0, mask.shape[0] - 1.0))
                bottom = float(np.clip(bottom, 0.0, mask.shape[0] - 1.0))
                if bottom <= top:
                    continue
                raw.append(
                    VerticalSegment(
                        scan_x_px=float(center_x),
                        y_top_px=top,
                        y_bottom_px=bottom,
                        source_black_component_id=int(source_component_id),
                        scan_column_index=int(grid_column_index),
                    )
                )

    return order_vertical_segments(raw)


def draw_capsule(mask: np.ndarray, a: tuple[float, float], b: tuple[float, float], radius_px: float) -> None:
    p0 = (int(round(a[0])), int(round(a[1])))
    p1 = (int(round(b[0])), int(round(b[1])))
    radius = max(1, int(round(radius_px)))
    thickness = max(1, 2 * radius)
    cv2.line(mask, p0, p1, 255, thickness, cv2.LINE_8)
    cv2.circle(mask, p0, radius, 255, -1, cv2.LINE_8)
    cv2.circle(mask, p1, radius, 255, -1, cv2.LINE_8)


def render_segment_mask(shape: tuple[int, int], segments: list[VerticalSegment], pen_diameter_px: float) -> np.ndarray:
    rendered = np.zeros(shape, dtype=np.uint8)
    radius = 0.5 * float(pen_diameter_px)
    for segment in segments:
        draw_capsule(
            rendered,
            (segment.scan_x_px, segment.y_top_px),
            (segment.scan_x_px, segment.y_bottom_px),
            radius,
        )
    return rendered


def add_coverage_repair_segments(
    black_mask: np.ndarray,
    source_labels: np.ndarray,
    segments: list[VerticalSegment],
    pen_diameter_px: float,
    minimum_centerline_length_px: float,
    maximum_iterations: int = 6,
) -> tuple[list[VerticalSegment], int]:
    repaired = list(segments)
    added = 0
    radius_px = 0.5 * float(pen_diameter_px)
    for iteration in range(maximum_iterations):
        rendered = render_segment_mask(black_mask.shape, repaired, pen_diameter_px) > 0
        missed = (black_mask > 0) & ~rendered
        if not missed.any():
            break
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            missed.astype(np.uint8), connectivity=8
        )
        new_segments: list[VerticalSegment] = []
        for missed_id in range(1, count):
            component = labels == missed_id
            if not component.any():
                continue
            x_center = float(centroids[missed_id, 0])
            y_min = float(stats[missed_id, cv2.CC_STAT_TOP])
            y_max = y_min + float(stats[missed_id, cv2.CC_STAT_HEIGHT]) - 1.0
            top = y_min + radius_px
            bottom = y_max - radius_px
            if bottom <= top:
                center_y = 0.5 * (y_min + y_max)
                half = 0.5 * max(float(minimum_centerline_length_px), 0.1)
                top, bottom = center_y - half, center_y + half
            top = float(np.clip(top, 0.0, black_mask.shape[0] - 1.0))
            bottom = float(np.clip(bottom, 0.0, black_mask.shape[0] - 1.0))
            if bottom <= top:
                continue
            source_values = source_labels[component]
            source_values = source_values[source_values > 0]
            source_component_id = (
                int(np.bincount(source_values).argmax()) if source_values.size else 0
            )
            new_segments.append(
                VerticalSegment(
                    scan_x_px=x_center,
                    y_top_px=top,
                    y_bottom_px=bottom,
                    source_black_component_id=source_component_id,
                    scan_column_index=-(iteration + 1),
                )
            )
        if not new_segments:
            break
        repaired.extend(new_segments)
        added += len(new_segments)
        repaired = order_vertical_segments(repaired)
    return repaired, added


def render_previews(
    image: np.ndarray,
    black_mask: np.ndarray,
    segments: list[VerticalSegment],
    pen_diameter_px: float,
    output_dir: Path,
) -> dict[str, Any]:
    rendered = render_segment_mask(black_mask.shape, segments, pen_diameter_px)

    target = black_mask.astype(bool)
    drawn = rendered > 0
    covered = target & drawn
    missed = target & ~drawn
    overdraw = ~target & drawn
    target_count = int(target.sum())
    coverage_ratio = float(covered.sum() / max(1, target_count))

    black_mask_path = output_dir / "black_pixel_mask.png"
    cv2.imwrite(str(black_mask_path), np.where(target, 0, 255).astype(np.uint8))

    line_path = output_dir / "vertical_line_graph_preview.png"
    cv2.imwrite(str(line_path), np.where(drawn, 0, 255).astype(np.uint8))

    overlay = image.copy()
    overlay[target] = (35, 35, 35)
    overlay[covered] = (0, 170, 0)
    overlay[missed] = (0, 0, 255)
    overlay[overdraw] = (255, 180, 0)
    coverage_path = output_dir / "vertical_line_coverage_preview.png"
    cv2.imwrite(str(coverage_path), overlay)

    return {
        "black_pixel_count": target_count,
        "covered_black_pixel_count": int(covered.sum()),
        "missed_black_pixel_count": int(missed.sum()),
        "overdraw_pixel_count": int(overdraw.sum()),
        "black_pixel_coverage_ratio": coverage_ratio,
        "black_mask": str(black_mask_path),
        "line_preview": str(line_path),
        "coverage_preview": str(coverage_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_image", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--black-threshold", type=int, default=160)
    parser.add_argument("--minimum-component-size", type=int, default=1)
    parser.add_argument("--close-iterations", type=int, default=0)
    parser.add_argument("--paper-center-x", type=float, default=0.50)
    parser.add_argument("--paper-center-y", type=float, default=0.00)
    parser.add_argument("--paper-width", type=float, default=0.32)
    parser.add_argument("--paper-height", type=float, default=0.20)
    parser.add_argument("--paper-margin", type=float, default=0.01)
    parser.add_argument("--pen-diameter", type=float, default=0.002)
    parser.add_argument(
        "--line-overlap",
        type=float,
        default=0.50,
        help=(
            "Fractional overlap between neighboring pen-width vertical strips. "
            "Default 0.50 places centers one half pen diameter apart."
        ),
    )
    parser.add_argument(
        "--global-grid-phase",
        type=float,
        default=0.50,
        help="Phase of the globally aligned vertical raster within one center-spacing interval.",
    )
    parser.add_argument(
        "--coverage-repair-iterations",
        type=int,
        default=12,
        help="Maximum missed-black-pixel repair passes after dense raster construction.",
    )
    parser.add_argument(
        "--minimum-centerline-length",
        type=float,
        default=0.00005,
        help="Minimum centerline length in metres for black spots smaller than one pen diameter.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.black_threshold <= 255:
        parser.error("--black-threshold must be between 0 and 255")
    if args.minimum_component_size < 1:
        parser.error("--minimum-component-size must be at least 1")
    if args.close_iterations < 0:
        parser.error("--close-iterations cannot be negative")
    if args.pen_diameter <= 0:
        parser.error("--pen-diameter must be positive")
    if not 0.0 <= args.line_overlap < 0.95:
        parser.error("--line-overlap must be in [0, 0.95)")
    if not 0.0 <= args.global_grid_phase <= 1.0:
        parser.error("--global-grid-phase must be in [0, 1]")
    if args.coverage_repair_iterations < 0:
        parser.error("--coverage-repair-iterations cannot be negative")
    if args.minimum_centerline_length <= 0:
        parser.error("--minimum-centerline-length must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.input_image.expanduser().resolve()
    image = read_image_on_white(input_path)
    black_mask, labels, stats = make_black_mask(
        image,
        args.black_threshold,
        args.minimum_component_size,
        args.close_iterations,
    )
    paper_map = calculate_paper_map(
        black_mask,
        args.paper_center_x,
        args.paper_center_y,
        args.paper_width,
        args.paper_height,
        args.paper_margin,
    )
    pen_diameter_px = float(args.pen_diameter) / paper_map.scale_m_per_px
    minimum_centerline_length_px = float(args.minimum_centerline_length) / paper_map.scale_m_per_px
    segments = build_vertical_segments(
        black_mask,
        labels,
        stats,
        pen_diameter_px,
        args.line_overlap,
        minimum_centerline_length_px,
        args.global_grid_phase,
    )
    segments, coverage_repair_line_count = add_coverage_repair_segments(
        black_mask,
        labels,
        segments,
        pen_diameter_px,
        minimum_centerline_length_px,
        maximum_iterations=args.coverage_repair_iterations,
    )
    if not segments:
        raise RuntimeError("The black mask did not produce any vertical line segments")

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    column_rank_by_x = {x: i for i, x in enumerate(sorted({round(s.scan_x_px, 6) for s in segments}))}
    for line_id, segment in enumerate(segments, start=1):
        # The build function already alternates column ordering. Alternate edge
        # orientation by ordered column position to retain a serpentine raster.
        x_key = round(segment.scan_x_px, 6)
        column_rank = column_rank_by_x[x_key]
        top_to_bottom = column_rank % 2 == 0
        if top_to_bottom:
            source_y_px, target_y_px = segment.y_top_px, segment.y_bottom_px
        else:
            source_y_px, target_y_px = segment.y_bottom_px, segment.y_top_px
        source_x_m, source_y_m = paper_map.pixel_to_world(segment.scan_x_px, source_y_px)
        target_x_m, target_y_m = paper_map.pixel_to_world(segment.scan_x_px, target_y_px)
        source_node_id = 2 * line_id - 1
        target_node_id = 2 * line_id
        component_id = line_id
        for node_id, y_px, x_m, y_m, endpoint_name in (
            (source_node_id, source_y_px, source_x_m, source_y_m, "source"),
            (target_node_id, target_y_px, target_x_m, target_y_m, "target"),
        ):
            node_rows.append(
                {
                    "topo_node_id": node_id,
                    "component_id": component_id,
                    "line_id": line_id,
                    "source_black_component_id": segment.source_black_component_id,
                    "x_px": round(segment.scan_x_px, 6),
                    "y_px": round(y_px, 6),
                    "x_original_px": round(segment.scan_x_px, 6),
                    "y_original_px": round(y_px, 6),
                    "x_m": round(x_m, 9),
                    "y_m": round(y_m, 9),
                    "pixel_degree": 1,
                    "node_type": "endpoint",
                    "endpoint_role": endpoint_name,
                }
            )
        length_px = abs(target_y_px - source_y_px)
        length_m = math.hypot(target_x_m - source_x_m, target_y_m - source_y_m)
        edge_rows.append(
            {
                "topo_edge_id": line_id,
                "component_id": component_id,
                "line_id": line_id,
                "scan_order": line_id,
                "grid_column_index": int(segment.scan_column_index),
                "source_black_component_id": segment.source_black_component_id,
                "source_topo_node_id": source_node_id,
                "target_topo_node_id": target_node_id,
                "source_x_px": round(segment.scan_x_px, 6),
                "source_y_px": round(source_y_px, 6),
                "target_x_px": round(segment.scan_x_px, 6),
                "target_y_px": round(target_y_px, 6),
                "source_x_m": round(source_x_m, 9),
                "source_y_m": round(source_y_m, 9),
                "target_x_m": round(target_x_m, 9),
                "target_y_m": round(target_y_m, 9),
                "point_count": 2,
                "length_px": round(length_px, 6),
                "length_original_px": round(length_px, 6),
                "length_m": round(length_m, 9),
                "stroke_width_m": round(float(args.pen_diameter), 9),
                "stroke_width_px": round(pen_diameter_px, 6),
                "center_spacing_m": round(float(args.pen_diameter) * (1.0 - args.line_overlap), 9),
                "direction": "vertical",
                "is_cycle": 0,
                "geometry_type": "vertical_pen_width_line",
                "path_coordinates_px": json.dumps(
                    [[round(segment.scan_x_px, 6), round(source_y_px, 6)],
                     [round(segment.scan_x_px, 6), round(target_y_px, 6)]],
                    separators=(",", ":"),
                ),
                "path_coordinates_m": json.dumps(
                    [[round(source_x_m, 9), round(source_y_m, 9)],
                     [round(target_x_m, 9), round(target_y_m, 9)]],
                    separators=(",", ":"),
                ),
            }
        )

    node_fields = list(node_rows[0].keys())
    edge_fields = list(edge_rows[0].keys())
    nodes_path = output_dir / "vertical_line_nodes.csv"
    edges_path = output_dir / "vertical_line_edges.csv"
    write_csv(nodes_path, node_rows, node_fields)
    write_csv(edges_path, edge_rows, edge_fields)
    # Compatibility aliases keep downstream graph tooling simple.
    write_csv(output_dir / "outline_topology_nodes.csv", node_rows, node_fields)
    write_csv(output_dir / "outline_topology_edges.csv", edge_rows, edge_fields)

    preview_stats = render_previews(image, black_mask, segments, pen_diameter_px, output_dir)
    summary = {
        "build": BUILD,
        "input_image": str(input_path),
        "image_width_px": int(image.shape[1]),
        "image_height_px": int(image.shape[0]),
        "black_threshold": int(args.black_threshold),
        "minimum_component_size_px": int(args.minimum_component_size),
        "black_connected_components": int(len(stats) - 1),
        "vertical_line_count": int(len(edge_rows)),
        "coverage_repair_line_count": int(coverage_repair_line_count),
        "graph_node_count": int(len(node_rows)),
        "graph_edge_count": int(len(edge_rows)),
        "graph_contract": "one independent vertical line = two nodes + one edge",
        "paper_center_x_m": float(args.paper_center_x),
        "paper_center_y_m": float(args.paper_center_y),
        "paper_width_m": float(args.paper_width),
        "paper_height_m": float(args.paper_height),
        "paper_margin_m": float(args.paper_margin),
        "pixel_bounds": {
            "min_x": paper_map.min_x_px,
            "max_x": paper_map.max_x_px,
            "min_y": paper_map.min_y_px,
            "max_y": paper_map.max_y_px,
        },
        "world_scale_m_per_px": paper_map.scale_m_per_px,
        "pen_diameter_m": float(args.pen_diameter),
        "pen_diameter_px": pen_diameter_px,
        "line_overlap_fraction": float(args.line_overlap),
        "line_center_spacing_m": float(args.pen_diameter) * (1.0 - args.line_overlap),
        "global_grid_phase": float(args.global_grid_phase),
        "coverage_repair_iterations": int(args.coverage_repair_iterations),
        "sampling_mode": "globally_phase_locked_dense_vertical_grid",
        "neighbor_overlap_margin_m": float(args.pen_diameter) * float(args.line_overlap),
        "single_missing_column_residual_gap_m": max(
            0.0,
            2.0 * float(args.pen_diameter) * (1.0 - args.line_overlap)
            - float(args.pen_diameter),
        ),
        "path_geometry": "dense_vertical_pen_width_lines",
        **preview_stats,
    }
    summary_json = output_dir / "vertical_fill_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_csv = output_dir / "vertical_fill_summary.csv"
    flat_rows = []
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(",", ":"))
        flat_rows.append({"parameter": key, "value": value})
    write_csv(summary_csv, flat_rows, ["parameter", "value"])

    readme = output_dir / "README_vertical_line_graph.txt"
    readme.write_text(
        "BLACK PIXELS -> PEN-WIDTH VERTICAL LINE GRAPH\n\n"
        "vertical_line_nodes.csv and vertical_line_edges.csv are the primary graph files.\n"
        "Each line is an independent graph component with two endpoint nodes and one edge.\n"
        "stroke_width_m equals the physical pen diameter, while neighboring line centers\n"
        "are separated by pen_diameter * (1 - line_overlap).\n",
        encoding="utf-8",
    )
    bundle_path = output_dir / "vertical_line_graph_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (
            nodes_path,
            edges_path,
            output_dir / "outline_topology_nodes.csv",
            output_dir / "outline_topology_edges.csv",
            output_dir / "black_pixel_mask.png",
            output_dir / "vertical_line_graph_preview.png",
            output_dir / "vertical_line_coverage_preview.png",
            summary_json,
            summary_csv,
            readme,
        ):
            archive.write(path, arcname=path.name)

    if not args.quiet:
        print(f"GRAPH BUILD: {BUILD}")
        print(
            f"Black components={len(stats)-1}, vertical lines={len(edge_rows)}, "
            f"pen diameter={args.pen_diameter:.6f} m ({pen_diameter_px:.3f} px), "
            f"coverage={preview_stats['black_pixel_coverage_ratio']:.4f}"
        )
        print(f"Wrote {nodes_path}")
        print(f"Wrote {edges_path}")
        print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
