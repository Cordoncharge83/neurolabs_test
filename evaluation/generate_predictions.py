#!/usr/bin/env python3
"""Generate SKU predictions for holdout boxes and save them to predictions.csv."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.predict import load_prediction_context, predict_image

HOLDOUT_TO_PREDICT_PATH = REPO_ROOT / "holdout" / "holdout_to_predict.csv"
TEMPLATE_PATH = REPO_ROOT / "holdout" / "predictions_template.csv"
OUTPUT_PATH = REPO_ROOT / "predictions.csv"
HOLDOUT_IMAGES_DIR = REPO_ROOT / "data" / "holdout"
HOLDOUT_CROPS_DIR = REPO_ROOT / "data" / "holdout_crops"


def load_holdout_rows(csv_path: Path) -> pd.DataFrame:
    """Load the holdout box definitions from CSV."""
    return pd.read_csv(csv_path)


def crop_box(image: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """Crop an image using x, y, w, h coordinates."""
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be positive")

    left = int(x)
    top = int(y)
    right = left + int(w)
    bottom = top + int(h)
    return image.crop((left, top, right, bottom))


def generate_predictions() -> pd.DataFrame:
    """Create predictions for holdout boxes and save them to predictions.csv."""
    holdout_df = load_holdout_rows(HOLDOUT_TO_PREDICT_PATH)
    context = load_prediction_context()

    predictions: list[dict[str, Any]] = []
    failed = 0

    for index, row in enumerate(holdout_df.to_dict(orient="records"), start=1):
        image_id = str(row.get("image_id", "")).strip()
        box_id = str(row.get("box_id", "")).strip()
        image_path = HOLDOUT_IMAGES_DIR / f"{image_id}.jpg"

        if not image_path.exists():
            failed += 1
            print(f"Warning: missing source image for {image_id}: {image_path}")
            predictions.append({"box_id": box_id, "pred_sku_id": ""})
            continue

        try:
            with Image.open(image_path) as image:
                cropped = crop_box(image.convert("RGB"), row.get("x"), row.get("y"), row.get("w"), row.get("h"))
                crop_dir = HOLDOUT_CROPS_DIR / image_id
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / f"{box_id}.jpg"
                cropped.save(crop_path)
                pred_sku_id, _, _ = predict_image(context, crop_path)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as error:
            failed += 1
            print(f"Warning: failed for {image_id}/{box_id}: {error}")
            predictions.append({"box_id": box_id, "pred_sku_id": ""})
        else:
            predictions.append({"box_id": box_id, "pred_sku_id": pred_sku_id})

        if index % 100 == 0:
            print(f"Progress: processed {index} boxes")

    predictions_df = pd.DataFrame(predictions)
    if TEMPLATE_PATH.exists():
        template_df = pd.read_csv(TEMPLATE_PATH)
        if "box_id" in template_df.columns and "pred_sku_id" in template_df.columns:
            output_df = template_df[["box_id", "pred_sku_id"]].copy()
            output_df["pred_sku_id"] = predictions_df["pred_sku_id"].tolist()
            output_df.to_csv(OUTPUT_PATH, index=False)
            return output_df

    predictions_df.to_csv(OUTPUT_PATH, index=False)
    return predictions_df


def main() -> None:
    """Generate and save holdout predictions."""
    generate_predictions()
    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
