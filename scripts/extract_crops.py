#!/usr/bin/env python3
"""Create bounding-box crops from local shelf images based on ground truth rows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
CROPS_DIR = DATA_DIR / "crops"
GROUND_TRUTH_PATH = REPO_ROOT / "ground_truth.csv"


def ensure_directory(path: Path) -> None:
    """Create a directory and any missing parents."""
    path.mkdir(parents=True, exist_ok=True)


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """Load ground-truth rows from a CSV file."""
    return pd.read_csv(csv_path)


def clean_text(value: object) -> str:
    """Convert a CSV field to a trimmed string, treating missing values as empty."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_coordinate(value: object, *, column_name: str) -> int:
    """Parse a coordinate value into an integer."""
    if pd.isna(value):
        raise ValueError(f"missing {column_name}")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {column_name}: {value}") from exc

    if not math.isfinite(number):
        raise ValueError(f"invalid {column_name}: {value}")

    return int(number)


def crop_box(image: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """Crop an image using x, y, w, h and return the resulting image."""
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be positive")

    left = x
    top = y
    right = x + w
    bottom = y + h

    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError("invalid bounding box")

    return image.crop((left, top, right, bottom))


def process_ground_truth(rows: Iterable[dict[str, object]]) -> tuple[int, int, int]:
    """Create crop files for each ground-truth row and return summary counts."""
    ensure_directory(CROPS_DIR)

    created = 0
    skipped = 0
    failed = 0
    rows = list(rows)
    total_rows = len(rows)

    for index, row in enumerate(rows, start=1):
        image_id = clean_text(row.get("image_id"))
        box_id = clean_text(row.get("box_id"))

        if not image_id or not box_id:
            failed += 1
            print(f"[{index}/{total_rows}] invalid image_id or box_id, skipping")
            continue

        image_path = IMAGES_DIR / f"{image_id}.jpg"
        if not image_path.exists():
            failed += 1
            print(f"[{index}/{total_rows}] missing image for {image_id}, skipping")
            continue

        try:
            x = parse_coordinate(row.get("x"), column_name="x")
            y = parse_coordinate(row.get("y"), column_name="y")
            w = parse_coordinate(row.get("w"), column_name="w")
            h = parse_coordinate(row.get("h"), column_name="h")
        except ValueError as error:
            failed += 1
            print(f"[{index}/{total_rows}] invalid coordinates for {image_id}/{box_id}: {error}")
            continue

        output_dir = CROPS_DIR / image_id
        ensure_directory(output_dir)
        output_path = output_dir / f"{box_id}.jpg"
        if output_path.exists():
            skipped += 1
            print(f"[{index}/{total_rows}] skipping existing crop {output_path}")
            continue

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                cropped_image = crop_box(image, x, y, w, h)
                cropped_image.save(output_path)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as error:
            failed += 1
            print(f"[{index}/{total_rows}] failed for {image_id}/{box_id}: {error}")
        else:
            created += 1
            print(f"[{index}/{total_rows}] created {output_path}")

    return created, skipped, failed


def main() -> None:
    """Extract all crops from the ground-truth CSV into the local crops directory."""
    ensure_directory(DATA_DIR)
    ensure_directory(IMAGES_DIR)
    ensure_directory(CROPS_DIR)

    ground_truth_df = load_ground_truth(GROUND_TRUTH_PATH)
    created, skipped, failed = process_ground_truth(ground_truth_df.to_dict(orient="records"))

    print("\nCrop summary")
    print(f"- Crops created: {created}")
    print(f"- Crops skipped: {skipped}")
    print(f"- Failed crops: {failed}")


if __name__ == "__main__":
    main()
