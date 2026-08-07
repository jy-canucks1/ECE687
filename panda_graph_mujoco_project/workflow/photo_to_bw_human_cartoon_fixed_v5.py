#!/usr/bin/env python3
"""
Convert a real human photograph into a simplified black-and-white cartoon
without training a new model. This revision does not require
cv2.CascadeClassifier and falls back safely when it is unavailable.

Default pipeline:
    pretrained DeepLabV3 person segmentation
    + portrait-aware region-boundary simplification
    + silhouette preservation
    + connected-component cleanup

The pretrained network is used only for inference. No training dataset,
epochs, or GAN training are required.

Requirements:
    pip install torch torchvision opencv-python pillow numpy

Example:
    python3 photo_to_bw_human_cartoon.py input.jpg -o output.png

For a full-body player against a complicated background:
    python3 photo_to_bw_human_cartoon.py player.jpg -o player_cartoon.png \
        --people largest --render-mode auto --portrait-style ink --detail medium --line-width 1 --save-debug

For multiple people:
    python3 photo_to_bw_human_cartoon.py team.jpg -o team_cartoon.png \
        --people all
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a simplified black-and-white human cartoon from a "
            "photograph using pretrained segmentation and classical vision."
        )
    )
    parser.add_argument("input", type=Path, help="Input image or directory.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output image or output directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, and so on.",
    )
    parser.add_argument(
        "--people",
        choices=("largest", "all"),
        default="largest",
        help=(
            "Keep only the largest detected person or all detected people. "
            "Use 'largest' for a main soccer-player photograph."
        ),
    )
    parser.add_argument(
        "--segmentation-threshold",
        type=float,
        default=0.35,
        help="Person probability threshold in [0, 1].",
    )
    parser.add_argument(
        "--detail",
        choices=("low", "medium", "high"),
        default="medium",
        help="Amount of internal face, clothing, and body detail.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("auto", "portrait", "outline"),
        default="auto",
        help=(
            "Rendering strategy. 'portrait' simplifies close-up faces into "
            "large readable features; 'outline' keeps the original "
            "multiscale-edge method; 'auto' selects portrait mode when a "
            "person fills most of the frame."
        ),
    )
    parser.add_argument(
        "--portrait-style",
        choices=("ink", "line"),
        default="ink",
        help=(
            "Close-up portrait appearance. 'ink' keeps simplified filled "
            "black facial features and is more readable; 'line' converts "
            "those regions to contour outlines for graph-oriented output."
        ),
    )
    parser.add_argument(
        "--line-width",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Final black-line width.",
    )
    parser.add_argument(
        "--minimum-component",
        type=int,
        default=14,
        help="Minimum non-face edge component size.",
    )
    parser.add_argument(
        "--face-detail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use lower local thresholds in detected face regions.",
    )
    parser.add_argument(
        "--include-near-body",
        type=int,
        default=3,
        help=(
            "Include internal lines this many pixels outside the person mask. "
            "Useful for hair and loose clothing."
        ),
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1600,
        help=(
            "Downscale the longest image side before processing. "
            "Use 0 to preserve the original size."
        ),
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save segmentation masks and intermediate edge maps.",
    )
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    return device


def image_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported input image: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(path)

    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No supported images were found in {path}")
    return files


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def resize_longest_side(
    image: np.ndarray,
    maximum_size: int,
) -> tuple[np.ndarray, float]:
    if maximum_size <= 0:
        return image, 1.0

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= maximum_size:
        return image, 1.0

    scale = maximum_size / float(longest)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count <= 1:
        return np.zeros_like(mask)

    largest_label = 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes without accidentally flooding the foreground.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(
        binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    flood = padded.copy()
    flood_mask = np.zeros(
        (padded.shape[0] + 2, padded.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)
    flood = flood[1:-1, 1:-1]
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes)


class PersonSegmenter:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.model = deeplabv3_mobilenet_v3_large(
            weights=self.weights
        ).to(device)
        self.model.eval()
        self.preprocess = self.weights.transforms()

        categories = list(self.weights.meta["categories"])
        try:
            self.person_index = categories.index("person")
        except ValueError as error:
            raise RuntimeError(
                "The loaded segmentation weights do not contain a 'person' class."
            ) from error

    @torch.inference_mode()
    def probability(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)["out"]
        probabilities = torch.softmax(logits, dim=1)
        person = probabilities[:, self.person_index : self.person_index + 1]

        person = F.interpolate(
            person,
            size=image_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return person.cpu().numpy()


def refine_person_mask(
    probability: np.ndarray,
    threshold: float,
    people_mode: str,
) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--segmentation-threshold must be between 0 and 1.")

    mask = np.where(probability >= threshold, 255, 0).astype(np.uint8)

    height, width = mask.shape
    scale = max(1, round(min(height, width) / 350))
    close_size = max(3, 2 * scale + 1)
    open_size = max(3, 2 * max(1, scale // 2) + 1)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), dtype=np.uint8),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((open_size, open_size), dtype=np.uint8),
        iterations=1,
    )
    mask = fill_holes(mask)

    if people_mode == "largest":
        mask = keep_largest_connected_component(mask)

    return mask


def estimate_face_regions_from_person_mask(
    person_mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """
    Estimate probable face/head regions from person-mask geometry.

    Close-up portraits need a substantially larger region than full-body
    photographs. The previous fallback used a narrow box around the upper
    center of the person, which caused most eyes, cheeks, ears, beard and hair
    to be processed by the noisy global edge detector.
    """
    binary = (person_mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    image_height, image_width = person_mask.shape
    image_area = image_height * image_width
    minimum_area = max(200, round(image_area * 0.0008))
    regions: list[tuple[int, int, int, int]] = []

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if width < 12 or height < 20:
            continue

        width_fraction = width / max(1, image_width)
        height_fraction = height / max(1, image_height)
        closeup = (
            width_fraction >= 0.62
            and height_fraction >= 0.68
            and y <= round(0.12 * image_height)
        )

        if closeup:
            # Include ears, jaw, beard and most of the hair mass. The lower
            # part may overlap the neck, which is preferable to cutting off
            # the chin or beard.
            estimated_width = min(
                width,
                max(48, int(round(0.88 * width))),
            )
            estimated_height = min(
                height,
                max(60, int(round(0.84 * height))),
            )
            center_x = x + width // 2
            face_x = center_x - estimated_width // 2
            face_y = y + int(round(0.035 * height))
        else:
            # Full-body or upper-body photograph: estimate a conventional
            # upper-center head box.
            estimated_width = int(round(min(0.50 * width, 0.32 * height)))
            estimated_width = max(24, estimated_width)
            estimated_width = min(estimated_width, width)

            estimated_height = int(round(1.25 * estimated_width))
            estimated_height = min(
                estimated_height,
                max(24, int(round(0.36 * height))),
            )
            estimated_height = min(estimated_height, height)

            center_x = x + width // 2
            face_x = center_x - estimated_width // 2
            face_y = y + max(0, int(round(0.012 * height)))

        face_x = max(0, min(face_x, image_width - estimated_width))
        face_y = max(0, min(face_y, image_height - estimated_height))
        regions.append(
            (
                int(face_x),
                int(face_y),
                int(estimated_width),
                int(estimated_height),
            )
        )

    return regions

def find_haar_cascade_file() -> Path | None:
    """
    Return a readable frontal-face Haar cascade path, or None.

    OpenCV can expose cv2.data.haarcascades even when the XML file itself was
    not installed. Constructing CascadeClassifier with that missing path makes
    OpenCV print a C++ persistence error before Python can catch anything, so
    the file must be verified before CascadeClassifier is created.
    """
    candidates: list[Path] = []

    cv2_data = getattr(cv2, "data", None)
    haar_directory = getattr(cv2_data, "haarcascades", None)
    if haar_directory:
        candidates.append(
            Path(haar_directory) / "haarcascade_frontalface_default.xml"
        )

    candidates.extend(
        [
            Path(
                "/usr/share/opencv4/haarcascades/"
                "haarcascade_frontalface_default.xml"
            ),
            Path(
                "/usr/share/opencv/haarcascades/"
                "haarcascade_frontalface_default.xml"
            ),
        ]
    )

    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue

    return None


def detect_faces(
    gray: np.ndarray,
    person_mask: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], str]:
    """
    Detect faces with a Haar cascade only when both the classifier and a valid
    XML file are available. Otherwise, use the person-mask head estimator.
    """
    cascade_class = getattr(cv2, "CascadeClassifier", None)
    cascade_file = find_haar_cascade_file()

    if cascade_class is not None and cascade_file is not None:
        try:
            cascade = cascade_class(str(cascade_file))
            if not cascade.empty():
                minimum = max(40, round(min(gray.shape[:2]) * 0.06))
                detected = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=5,
                    minSize=(minimum, minimum),
                )
                faces = [
                    tuple(int(value) for value in face)
                    for face in detected
                ]
                if faces:
                    return faces, "opencv-haar"
        except (AttributeError, cv2.error, OSError, TypeError):
            pass

    return (
        estimate_face_regions_from_person_mask(person_mask),
        "person-mask-fallback",
    )


def luminance_channels(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray = lab[:, :, 0]

    tile = max(4, round(min(gray.shape[:2]) / 100))
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(tile, tile),
    )
    enhanced = clahe.apply(gray)
    return gray, enhanced


def detail_parameters(detail: str) -> dict[str, float | int]:
    if detail == "low":
        return {
            "strong_low": 65,
            "strong_high": 155,
            "medium_low": 45,
            "medium_high": 125,
            "coarse_low": 35,
            "coarse_high": 100,
            "medium_sigma": 1.7,
            "coarse_sigma": 3.0,
        }
    if detail == "high":
        return {
            "strong_low": 35,
            "strong_high": 95,
            "medium_low": 24,
            "medium_high": 70,
            "coarse_low": 16,
            "coarse_high": 52,
            "medium_sigma": 1.1,
            "coarse_sigma": 2.0,
        }
    return {
        "strong_low": 48,
        "strong_high": 125,
        "medium_low": 32,
        "medium_high": 92,
        "coarse_low": 24,
        "coarse_high": 72,
        "medium_sigma": 1.4,
        "coarse_sigma": 2.5,
    }


def multiscale_edges(
    enhanced: np.ndarray,
    detail: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = detail_parameters(detail)

    bilateral = cv2.bilateralFilter(
        enhanced,
        d=9,
        sigmaColor=45,
        sigmaSpace=45,
    )
    strong = cv2.Canny(
        bilateral,
        int(params["strong_low"]),
        int(params["strong_high"]),
        L2gradient=True,
    )

    medium_image = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        sigmaX=float(params["medium_sigma"]),
    )
    medium = cv2.Canny(
        medium_image,
        int(params["medium_low"]),
        int(params["medium_high"]),
        L2gradient=True,
    )

    coarse_image = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        sigmaX=float(params["coarse_sigma"]),
    )
    coarse = cv2.Canny(
        coarse_image,
        int(params["coarse_low"]),
        int(params["coarse_high"]),
        L2gradient=True,
    )
    return strong, medium, coarse


def face_mask(
    shape: tuple[int, int],
    faces: Iterable[tuple[int, int, int, int]],
    padding_ratio: float = 0.14,
) -> np.ndarray:
    height, width = shape
    output = np.zeros((height, width), dtype=np.uint8)

    for x, y, face_width, face_height in faces:
        pad_x = round(face_width * padding_ratio)
        pad_y = round(face_height * padding_ratio)

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(width, x + face_width + pad_x)
        y1 = min(height, y + face_height + pad_y)
        output[y0:y1, x0:x1] = 255

    return output


def add_face_edges(
    edge_map: np.ndarray,
    enhanced: np.ndarray,
    faces: Iterable[tuple[int, int, int, int]],
) -> np.ndarray:
    result = edge_map.copy()
    height, width = edge_map.shape

    for x, y, face_width, face_height in faces:
        pad_x = round(face_width * 0.12)
        pad_y = round(face_height * 0.12)

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(width, x + face_width + pad_x)
        y1 = min(height, y + face_height + pad_y)

        region = enhanced[y0:y1, x0:x1]
        if region.size == 0:
            continue

        region = cv2.bilateralFilter(
            region,
            d=7,
            sigmaColor=30,
            sigmaSpace=30,
        )

        median = float(np.median(region))
        lower = max(18, round(0.45 * median))
        upper = max(lower + 20, min(130, round(1.10 * median)))

        local_edges = cv2.Canny(
            region,
            lower,
            upper,
            L2gradient=True,
        )

        # Replace the face region instead of blindly adding every global edge.
        # This limits duplicate cheek and shadow contours.
        result[y0:y1, x0:x1] = cv2.bitwise_or(
            result[y0:y1, x0:x1],
            local_edges,
        )

    return result


def largest_person_bbox(
    person_mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    binary = (person_mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (
        int(stats[label, cv2.CC_STAT_LEFT]),
        int(stats[label, cv2.CC_STAT_TOP]),
        int(stats[label, cv2.CC_STAT_WIDTH]),
        int(stats[label, cv2.CC_STAT_HEIGHT]),
    )


def choose_render_mode(
    requested_mode: str,
    person_mask: np.ndarray,
    faces: Iterable[tuple[int, int, int, int]],
) -> str:
    if requested_mode != "auto":
        return requested_mode

    image_height, image_width = person_mask.shape
    bbox = largest_person_bbox(person_mask)
    if bbox is None:
        return "outline"

    x, y, width, height = bbox
    bbox_area_fraction = (width * height) / max(
        1.0, float(image_height * image_width)
    )
    closeup_geometry = (
        width >= 0.68 * image_width
        and height >= 0.72 * image_height
        and y <= 0.12 * image_height
        and bbox_area_fraction >= 0.50
    )

    large_face = any(
        face_width >= 0.42 * image_width
        and face_height >= 0.42 * image_height
        for _, _, face_width, face_height in faces
    )
    return "portrait" if closeup_geometry or large_face else "outline"


def remove_binary_components(
    binary: np.ndarray,
    minimum_area: int,
    maximum_area: int | None = None,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8),
        connectivity=8,
    )
    output = np.zeros_like(binary)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        if maximum_area is not None and area > maximum_area:
            continue
        output[labels == label] = 255
    return output


def contour_boundaries_from_regions(
    regions: np.ndarray,
    minimum_perimeter: float,
    line_width: int = 1,
) -> np.ndarray:
    """
    Convert broad dark feature regions into simplified contour lines.

    This is much more stable for portraits than tracing every local grayscale
    gradient. Eyes, eyebrows, lips, nostrils, hair masses and beard masses are
    represented by their region boundaries rather than by individual hairs,
    skin pores or illumination texture.
    """
    contours, hierarchy = cv2.findContours(
        regions,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    output = np.zeros_like(regions)
    if hierarchy is None:
        return output

    hierarchy = hierarchy[0]
    for index, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        if perimeter < minimum_perimeter:
            continue

        area = abs(cv2.contourArea(contour))
        if area < 2.0:
            continue

        # Slightly stronger approximation for inner holes to avoid jagged
        # duplicate rings inside pupils, beard texture and highlighted hair.
        parent = int(hierarchy[index][3])
        epsilon_ratio = 0.006 if parent >= 0 else 0.004
        approximation = cv2.approxPolyDP(
            contour,
            epsilon_ratio * perimeter,
            True,
        )
        cv2.drawContours(
            output,
            [approximation],
            -1,
            255,
            line_width,
            cv2.LINE_AA,
        )
    return output


def simplified_portrait_edges(
    gray: np.ndarray,
    person_mask: np.ndarray,
    faces: Iterable[tuple[int, int, int, int]],
    detail: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Render readable close-up facial features.

    The image is intentionally reduced before thresholding. This removes
    pores, individual beard hairs and tiny hair highlights that made the old
    result illegible. Adaptive dark-feature regions are then converted to
    boundaries and merged with a conservative Canny map.
    """
    image_height, image_width = gray.shape
    face_list = list(faces)

    if face_list:
        # Prefer the largest face/head estimate.
        x, y, face_width, face_height = max(
            face_list,
            key=lambda item: item[2] * item[3],
        )
    else:
        bbox = largest_person_bbox(person_mask)
        if bbox is None:
            return (
                np.zeros_like(gray),
                np.zeros_like(gray),
                np.zeros_like(gray),
                np.zeros_like(gray),
            )
        x, y, face_width, face_height = bbox

    pad_x = int(round(0.04 * face_width))
    pad_y = int(round(0.03 * face_height))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image_width, x + face_width + pad_x)
    y1 = min(image_height, y + face_height + pad_y)

    crop = gray[y0:y1, x0:x1]
    crop_mask = person_mask[y0:y1, x0:x1]
    if crop.size == 0:
        return (
            np.zeros_like(gray),
            np.zeros_like(gray),
            np.zeros_like(gray),
            np.zeros_like(gray),
        )

    # Process at a controlled spatial scale. A 300-pixel face does not need
    # 300 pixels of texture to represent recognizable eyes and mouth.
    target_width = {
        "low": 155,
        "medium": 190,
        "high": 230,
    }[detail]
    scale = min(1.0, target_width / max(1.0, float(crop.shape[1])))
    small_size = (
        max(32, int(round(crop.shape[1] * scale))),
        max(32, int(round(crop.shape[0] * scale))),
    )
    small = cv2.resize(
        crop,
        small_size,
        interpolation=cv2.INTER_AREA,
    )
    small_mask = cv2.resize(
        crop_mask,
        small_size,
        interpolation=cv2.INTER_NEAREST,
    )

    # Repeated edge-preserving smoothing suppresses skin texture while
    # retaining major feature boundaries.
    smooth = cv2.bilateralFilter(small, 9, 55, 55)
    smooth = cv2.bilateralFilter(smooth, 9, 45, 45)
    smooth = cv2.medianBlur(smooth, 5)

    block_fraction = {
        "low": 0.14,
        "medium": 0.115,
        "high": 0.095,
    }[detail]
    block_size = int(round(min(smooth.shape) * block_fraction))
    block_size = max(15, min(41, block_size))
    if block_size % 2 == 0:
        block_size += 1

    adaptive_c = {
        "low": 10,
        "medium": 8,
        "high": 6,
    }[detail]
    dark_regions = cv2.adaptiveThreshold(
        smooth,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        adaptive_c,
    )
    dark_regions = cv2.bitwise_and(dark_regions, small_mask)

    # Merge nearby hairs and beard strokes into readable masses before
    # extracting their outlines.
    dark_regions = cv2.morphologyEx(
        dark_regions,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    dark_regions = cv2.morphologyEx(
        dark_regions,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1,
    )

    crop_area = dark_regions.shape[0] * dark_regions.shape[1]
    minimum_region = {
        "low": max(10, int(round(crop_area * 0.00055))),
        "medium": max(7, int(round(crop_area * 0.00038))),
        "high": max(5, int(round(crop_area * 0.00024))),
    }[detail]
    dark_regions = remove_binary_components(
        dark_regions,
        minimum_area=minimum_region,
        maximum_area=int(round(0.45 * crop_area)),
    )

    region_edges = contour_boundaries_from_regions(
        dark_regions,
        minimum_perimeter=max(8.0, 0.025 * min(smooth.shape)),
        line_width=1,
    )

    canny_parameters = {
        "low": (70, 165),
        "medium": (58, 145),
        "high": (46, 125),
    }[detail]
    conservative = cv2.GaussianBlur(smooth, (0, 0), sigmaX=1.35)
    conservative = cv2.Canny(
        conservative,
        canny_parameters[0],
        canny_parameters[1],
        L2gradient=True,
    )
    conservative = cv2.bitwise_and(conservative, small_mask)
    conservative = remove_binary_components(
        conservative,
        minimum_area={
            "low": 12,
            "medium": 9,
            "high": 6,
        }[detail],
    )

    combined_small = cv2.bitwise_or(region_edges, conservative)

    # Resize line maps separately with nearest-neighbor interpolation to avoid
    # gray antialiased pixels entering the binary output.
    combined_crop = cv2.resize(
        combined_small,
        (crop.shape[1], crop.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    regions_crop = cv2.resize(
        region_edges,
        (crop.shape[1], crop.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    canny_crop = cv2.resize(
        conservative,
        (crop.shape[1], crop.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    ink_crop = cv2.resize(
        dark_regions,
        (crop.shape[1], crop.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    combined = np.zeros_like(gray)
    region_debug = np.zeros_like(gray)
    canny_debug = np.zeros_like(gray)
    ink_debug = np.zeros_like(gray)
    combined[y0:y1, x0:x1] = combined_crop
    region_debug[y0:y1, x0:x1] = regions_crop
    canny_debug[y0:y1, x0:x1] = canny_crop
    ink_debug[y0:y1, x0:x1] = ink_crop
    return combined, region_debug, canny_debug, ink_debug


def silhouette_edges(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(
        mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    return cv2.subtract(mask, eroded)


def filter_components(
    edges: np.ndarray,
    faces_mask: np.ndarray,
    minimum_component: int,
    render_mode: str,
) -> np.ndarray:
    binary = (edges > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    result = np.zeros_like(edges)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        center_x, center_y = centroids[label]
        center_x = int(round(center_x))
        center_y = int(round(center_y))

        in_face = (
            0 <= center_y < faces_mask.shape[0]
            and 0 <= center_x < faces_mask.shape[1]
            and faces_mask[center_y, center_x] > 0
        )

        if in_face:
            # The previous threshold of three pixels retained pores, tiny hair
            # fragments and beard speckles. Portrait mode uses a stricter
            # threshold because the important facial regions have already
            # been simplified before this stage.
            required = max(
                6 if render_mode == "portrait" else 4,
                minimum_component // (2 if render_mode == "portrait" else 3),
            )
        else:
            required = minimum_component

        if area >= required:
            result[labels == label] = 255

    return result

def make_cartoon(
    image_bgr: np.ndarray,
    probability: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mask = refine_person_mask(
        probability,
        threshold=args.segmentation_threshold,
        people_mode=args.people,
    )

    if np.count_nonzero(mask) == 0:
        raise RuntimeError(
            "No person was detected. Try a lower "
            "--segmentation-threshold such as 0.20."
        )

    gray, enhanced = luminance_channels(image_bgr)
    if args.face_detail:
        faces, face_detector = detect_faces(gray, mask)
    else:
        faces, face_detector = [], "disabled"
    faces_mask = face_mask(gray.shape, faces, padding_ratio=0.04)

    render_mode = choose_render_mode(
        args.render_mode,
        person_mask=mask,
        faces=faces,
    )

    strong, medium, coarse = multiscale_edges(
        enhanced,
        detail=args.detail,
    )

    body_gate = mask
    if args.include_near_body > 0:
        body_gate = cv2.dilate(
            body_gate,
            np.ones(
                (
                    2 * args.include_near_body + 1,
                    2 * args.include_near_body + 1,
                ),
                dtype=np.uint8,
            ),
            iterations=1,
        )

    strong = cv2.bitwise_and(strong, body_gate)

    interior = cv2.erode(
        mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    medium = cv2.bitwise_and(medium, interior)
    coarse = cv2.bitwise_and(coarse, interior)

    portrait_edges = np.zeros_like(gray)
    portrait_regions = np.zeros_like(gray)
    portrait_canny = np.zeros_like(gray)
    portrait_ink = np.zeros_like(gray)

    if render_mode == "portrait":
        portrait_edges, portrait_regions, portrait_canny, portrait_ink = (
            simplified_portrait_edges(
                gray,
                person_mask=mask,
                faces=faces,
                detail=args.detail,
            )
        )

        # Outside the face/head region, retain only conservative strong edges.
        # Inside it, replace the noisy multiscale result with the simplified
        # portrait renderer.
        outside_face = cv2.bitwise_not(faces_mask)
        body_edges = cv2.bitwise_and(strong, outside_face)
        portrait_render = (
            portrait_ink
            if args.portrait_style == "ink"
            else portrait_edges
        )
        internal = cv2.bitwise_or(body_edges, portrait_render)
    else:
        if args.detail == "low":
            internal = cv2.bitwise_or(strong, coarse)
        elif args.detail == "high":
            internal = cv2.bitwise_or(
                strong,
                cv2.bitwise_or(medium, coarse),
            )
        else:
            internal = cv2.bitwise_or(strong, medium)

        if args.face_detail and faces:
            internal = add_face_edges(internal, enhanced, faces)

    silhouette = silhouette_edges(mask)
    combined = cv2.bitwise_or(internal, silhouette)

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1,
    )

    combined = filter_components(
        combined,
        faces_mask=faces_mask,
        minimum_component=args.minimum_component,
        render_mode=render_mode,
    )

    combined = cv2.bitwise_or(combined, silhouette)

    if args.line_width > 1:
        kernel_size = 2 * args.line_width - 1
        combined = cv2.dilate(
            combined,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )

    black_on_white = cv2.bitwise_not(combined)

    overlay = image_bgr.copy()
    overlay[combined > 0] = (0, 0, 255)

    probability_preview = np.clip(
        probability * 255.0,
        0,
        255,
    ).astype(np.uint8)

    detector_debug = np.full(
        (58, max(560, image_bgr.shape[1]), 3),
        255,
        dtype=np.uint8,
    )
    cv2.putText(
        detector_debug,
        (
            f"Face detector: {face_detector}; regions: {len(faces)}; "
            f"render mode: {render_mode}; portrait style: {args.portrait_style}"
        ),
        (10, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    debug = {
        "00_face_detector_status.png": detector_debug,
        "01_person_probability.png": probability_preview,
        "02_person_mask.png": mask,
        "03_enhanced_luminance.png": enhanced,
        "04_strong_edges.png": strong,
        "05_medium_edges.png": medium,
        "06_coarse_edges.png": coarse,
        "07_face_regions.png": faces_mask,
        "08_silhouette.png": silhouette,
        "09_portrait_region_edges.png": portrait_regions,
        "10_portrait_conservative_canny.png": portrait_canny,
        "11_portrait_ink_regions.png": portrait_ink,
        "12_portrait_combined_lines.png": portrait_edges,
        "13_final_edges_or_ink.png": combined,
        "14_overlay.png": overlay,
    }
    return black_on_white, debug

def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def save_debug(
    output_path: Path,
    images: dict[str, np.ndarray],
) -> None:
    directory = output_path.parent / f"{output_path.stem}_debug"
    directory.mkdir(parents=True, exist_ok=True)
    for filename, image in images.items():
        write_image(directory / filename, image)


def output_path_for(
    source: Path,
    requested_output: Path,
    multiple_inputs: bool,
) -> Path:
    if multiple_inputs or requested_output.suffix == "":
        requested_output.mkdir(parents=True, exist_ok=True)
        return requested_output / f"{source.stem}_bw_cartoon.png"
    return requested_output


def main() -> None:
    args = parse_args()
    sources = image_files(args.input)
    device = choose_device(args.device)

    print(f"Device: {device}")
    print("Loading pretrained person-segmentation model...")
    segmenter = PersonSegmenter(device)
    print(f"Images: {len(sources)}")

    for index, source in enumerate(sources, start=1):
        image = load_bgr(source)
        image, scale = resize_longest_side(image, args.max_size)

        print(
            f"[{index:04d}/{len(sources):04d}] "
            f"{source.name} | size={image.shape[1]}x{image.shape[0]}"
        )

        probability = segmenter.probability(image)
        cartoon, debug = make_cartoon(image, probability, args)

        destination = output_path_for(
            source,
            args.output,
            multiple_inputs=len(sources) > 1,
        )
        write_image(destination, cartoon)

        if args.save_debug:
            save_debug(destination, debug)

        print(f"  Output: {destination}")

    print("Done.")


if __name__ == "__main__":
    main()
