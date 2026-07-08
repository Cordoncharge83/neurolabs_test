#!/usr/bin/env python3
"""Generate normalized CLIP embeddings for catalogue thumbnails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from clip_model import encode_image, load_clip_model

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_PATH = REPO_ROOT / "catalogue.csv"
THUMBS_DIR = REPO_ROOT / "data" / "thumbs"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "catalogue_embeddings.npy"
METADATA_PATH = EMBEDDINGS_DIR / "catalogue_metadata.csv"


def load_catalogue(csv_path: Path) -> pd.DataFrame:
    """Load catalogue rows from CSV."""
    return pd.read_csv(csv_path)


def build_embedding_rows(catalogue_df: pd.DataFrame) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Create embeddings and metadata for catalogue rows with available thumbnails."""
    model, preprocess, device = load_clip_model()

    embeddings: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    for index, row in catalogue_df.iterrows():
        sku_id = str(row.get("sku_id", "")).strip()
        if not sku_id:
            continue

        thumbnail_path = THUMBS_DIR / f"{sku_id}.jpg"
        if not thumbnail_path.exists():
            print(f"Warning: skipping {sku_id}: thumbnail not found at {thumbnail_path}")
            continue

        try:
            with Image.open(thumbnail_path) as image:
                image = image.convert("RGB")
                embedding = encode_image(model, preprocess, device, image)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as error:
            print(f"Warning: failed to embed {sku_id}: {error}")
            continue

        embeddings.append(embedding[0])
        metadata_rows.append(
            {
                "embedding_index": len(embeddings) - 1,
                "sku_id": sku_id,
                "name": row.get("name", ""),
                "brand": row.get("brand", ""),
                "flavour": row.get("flavour", ""),
                "size": row.get("size", ""),
            }
        )

    metadata_df = pd.DataFrame(metadata_rows)
    return embeddings, metadata_df


def save_embeddings(embeddings: list[np.ndarray], metadata_df: pd.DataFrame) -> None:
    """Save embeddings and metadata to disk."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if not embeddings:
        np.save(EMBEDDINGS_PATH, np.empty((0, 512), dtype=np.float32))
        metadata_df.to_csv(METADATA_PATH, index=False)
        return

    embedding_array = np.stack(embeddings, axis=0).astype(np.float32)
    np.save(EMBEDDINGS_PATH, embedding_array)
    metadata_df.to_csv(METADATA_PATH, index=False)


def main() -> None:
    """Generate and save CLIP embeddings for the catalogue thumbnails."""
    catalogue_df = load_catalogue(CATALOGUE_PATH)
    embeddings, metadata_df = build_embedding_rows(catalogue_df)
    save_embeddings(embeddings, metadata_df)

    print(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
