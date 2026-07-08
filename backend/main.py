#!/usr/bin/env python3
"""Minimal FastAPI backend for product recognition."""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.predict import load_prediction_context, predict_image

app = FastAPI(title="Product Recognition API")
DATA_IMAGES_DIR = REPO_ROOT / "data" / "images"
GROUND_TRUTH_PATH = REPO_ROOT / "ground_truth.csv"


@lru_cache(maxsize=1)
def get_prediction_context() -> Any:
    """Load the prediction context once and reuse it across requests."""
    return load_prediction_context()


class AnalyzeRequest(BaseModel):
    """Request payload for shelf image analysis."""

    image_id: str


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/images")
def list_images() -> dict[str, list[dict[str, str]]]:
    """Return available demo shelf images from the local data directory."""
    images = []
    for path in sorted(DATA_IMAGES_DIR.glob("*.jpg")):
        images.append({"image_id": path.stem, "filename": path.name})
    return {"images": images}


@app.post("/predict")
def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    """Predict the most similar catalogue SKU for an uploaded image."""
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file name provided")

    with tempfile.NamedTemporaryFile(suffix=Path(image.filename).suffix or ".jpg", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        content = image.file.read()
        temp_file.write(content)

    try:
        pred_sku_id, similarity, row = predict_image(get_prediction_context(), temp_path)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "pred_sku_id": pred_sku_id,
        "name": str(row.get("name", "")),
        "brand": str(row.get("brand", "")),
        "similarity": float(similarity),
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """Analyze a demo shelf image and return per-box recognition predictions."""
    image_id = request.image_id.strip()
    if not image_id:
        raise HTTPException(status_code=400, detail="image_id is required")

    image_path = DATA_IMAGES_DIR / f"{image_id}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    try:
        ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail="Ground truth data not found") from error

    matching_rows = ground_truth_df[ground_truth_df["image_id"] == image_id]
    valid_rows = []
    for row in matching_rows.to_dict(orient="records"):
        gt_sku_id = str(row.get("gt_sku_id", "")).strip()
        has_sku = str(row.get("gt_has_sku", "")).strip().lower() in {"true", "1", "yes", "y"}
        if not has_sku or not gt_sku_id:
            continue
        valid_rows.append(row)

    if not valid_rows:
        return {"image_id": image_id, "predictions": [], "summary": {"total_products": 0, "unique_skus": 0, "brand_counts": {}, "average_similarity": 0.0}}

    try:
        with Image.open(image_path) as shelf_image:
            shelf_image = shelf_image.convert("RGB")
            predictions: list[dict[str, Any]] = []
            similarities: list[float] = []
            brand_counts: Counter[str] = Counter()

            for row in valid_rows:
                try:
                    x = int(float(row.get("x", 0)))
                    y = int(float(row.get("y", 0)))
                    w = int(float(row.get("w", 0)))
                    h = int(float(row.get("h", 0)))
                except (TypeError, ValueError) as error:
                    print(f"Warning: invalid coordinates for {image_id}/{row.get('box_id', 'unknown')}: {error}")
                    continue

                if w <= 0 or h <= 0:
                    print(f"Warning: invalid crop size for {image_id}/{row.get('box_id', 'unknown')}")
                    continue

                crop = shelf_image.crop((x, y, x + w, y + h))
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                    crop.save(temp_path)

                try:
                    pred_sku_id, similarity, row_result = predict_image(get_prediction_context(), temp_path)
                except (FileNotFoundError, ValueError, OSError) as error:
                    print(f"Warning: prediction failed for {image_id}/{row.get('box_id', 'unknown')}: {error}")
                    temp_path.unlink(missing_ok=True)
                    continue
                finally:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError:
                        pass

                predictions.append(
                    {
                        "box_id": str(row.get("box_id", "")),
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "pred_sku_id": str(pred_sku_id),
                        "name": str(row_result.get("name", "")),
                        "brand": str(row_result.get("brand", "")),
                        "similarity": float(similarity),
                    }
                )
                similarities.append(float(similarity))
                brand_counts[str(row_result.get("brand", ""))] += 1
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    summary = {
        "total_products": len(predictions),
        "unique_skus": len({item["pred_sku_id"] for item in predictions if item.get("pred_sku_id")}),
        "brand_counts": dict(brand_counts),
        "average_similarity": round(sum(similarities) / len(similarities), 4) if similarities else 0.0,
    }

    return {"image_id": image_id, "predictions": predictions, "summary": summary}
