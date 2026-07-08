#!/usr/bin/env python3
"""Predict the most similar catalogue SKU for a single cropped product image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from recognition.clip_model import encode_image, load_clip_model

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = REPO_ROOT / "data" / "embeddings" / "catalogue_embeddings.npy"
METADATA_PATH = REPO_ROOT / "data" / "embeddings" / "catalogue_metadata.csv"


class PredictionContext(dict[str, Any]):
    """Container for model, preprocessing, and catalogue data used for prediction."""


def load_prediction_context() -> PredictionContext:
    """Load the CLIP model and catalogue data once for reuse across predictions."""
    embeddings = load_catalogue_embeddings(EMBEDDINGS_PATH)
    metadata = load_metadata(METADATA_PATH)
    model, preprocess, device = load_clip_model()

    context = PredictionContext(
        model=model,
        preprocess=preprocess,
        device=device,
        catalogue_embeddings=embeddings,
        catalogue_metadata=metadata,
    )
    return context


def load_catalogue_embeddings(embeddings_path: Path) -> np.ndarray:
    """Load and normalize catalogue embeddings from disk."""
    embeddings = np.load(embeddings_path)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load catalogue metadata from disk."""
    return pd.read_csv(metadata_path)


def predict_image(context: PredictionContext, image_path: Path) -> tuple[str, float, pd.Series]:
    """Predict the best matching SKU for an input crop image using a reusable context."""
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    embeddings = context["catalogue_embeddings"]
    metadata = context["catalogue_metadata"]
    model = context["model"]
    preprocess = context["preprocess"]
    device = context["device"]

    if embeddings.size == 0:
        raise ValueError("no catalogue embeddings available")
    if metadata.empty:
        raise ValueError("no catalogue metadata available")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            query_embedding = encode_image(model, preprocess, device, image)
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"could not read image: {error}") from error

    query_embedding = query_embedding[0]
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        raise ValueError("query embedding is empty")

    normalized_query = query_embedding / query_norm
    similarities = embeddings @ normalized_query
    best_index = int(np.argmax(similarities))
    best_similarity = float(similarities[best_index])
    best_row = metadata.iloc[best_index]
    return str(best_row["sku_id"]), best_similarity, best_row


def predict_sku(image_path: Path, embeddings: np.ndarray, metadata: pd.DataFrame) -> tuple[str, float, pd.Series]:
    """Predict the best matching SKU for an input crop image."""
    context = PredictionContext(
        model=load_clip_model()[0],
        preprocess=load_clip_model()[1],
        device=load_clip_model()[2],
        catalogue_embeddings=embeddings,
        catalogue_metadata=metadata,
    )
    return predict_image(context, image_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Predict the matching SKU for a cropped product image")
    parser.add_argument("image_path", help="Path to the cropped product image")
    return parser.parse_args()


def main() -> None:
    """Run SKU prediction for a single cropped image."""
    args = parse_args()
    image_path = Path(args.image_path).expanduser().resolve()

    context = load_prediction_context()
    sku_id, similarity, row = predict_image(context, image_path)

    print("Prediction")
    print(f"- predicted sku_id: {sku_id}")
    print(f"- name: {row.get('name', '')}")
    print(f"- brand: {row.get('brand', '')}")
    print(f"- similarity score: {similarity:.4f}")


if __name__ == "__main__":
    main()
