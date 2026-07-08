#!/usr/bin/env python3
"""Download dataset images and thumbnails into the local data directory."""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
THUMBS_DIR = DATA_DIR / "thumbs"
HOLDOUT_DIR = DATA_DIR / "holdout"


def ensure_directory(path: Path) -> None:
    """Create a directory and any missing parents."""
    path.mkdir(parents=True, exist_ok=True)


def load_csv_rows(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file using pandas and return its rows as a DataFrame."""
    return pd.read_csv(csv_path, dtype=str)


def infer_extension(url: str, default: str = ".jpg") -> str:
    """Infer a filename extension from the URL or fall back to a default."""
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    return suffix if suffix else default


def download_url(url: str, destination: Path) -> None:
    """Download a single URL into the destination path."""
    with urllib.request.urlopen(url, timeout=30) as response, open(destination, "wb") as handle:
        handle.write(response.read())


def is_valid_url(url: str | None) -> bool:
    """Return True when a URL looks like a usable HTTP(S) URL."""
    if not isinstance(url, str):
        return False

    value = url.strip()
    if not value:
        return False

    if value.lower() == "nan":
        return False

    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return False

    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def download_records(
    rows: Iterable[dict[str, str]],
    *,
    id_column: str,
    url_column: str,
    output_dir: Path,
    label: str,
) -> tuple[int, int, int]:
    """Download records from a collection of row dictionaries into a directory."""
    ensure_directory(output_dir)
    downloaded = 0
    skipped = 0
    failed = 0
    total_rows = len(list(rows))

    rows = list(rows)

    for index, row in enumerate(rows, start=1):
        record_id = str(row.get(id_column, "")).strip()
        raw_url = row.get(url_column)
        url = str(raw_url).strip() if raw_url is not None else ""

        if not record_id or not is_valid_url(url):
            print(f"[{label}] {index}: invalid identifier or URL, skipping")
            failed += 1
            continue

        destination = output_dir / f"{record_id}{infer_extension(url)}"
        if destination.exists():
            skipped += 1
            print(f"[{label}] {index}/{total_rows}: skipping existing {destination.name}")
            continue

        print(f"[{label}] {index}/{total_rows}: downloading {record_id} -> {destination.name}")
        try:
            download_url(url, destination)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
            failed += 1
            print(f"[{label}] {index}/{total_rows}: failed for {record_id}: {error}")
        else:
            downloaded += 1
            print(f"[{label}] {index}/{total_rows}: saved {destination.name}")

    return downloaded, skipped, failed


def main() -> None:
    """Download all shelf images, thumbnails, and holdout images into local data folders."""
    ensure_directory(DATA_DIR)
    ensure_directory(IMAGES_DIR)
    ensure_directory(THUMBS_DIR)
    ensure_directory(HOLDOUT_DIR)

    images_df = load_csv_rows(REPO_ROOT / "images.csv")
    catalogue_df = load_csv_rows(REPO_ROOT / "catalogue.csv")
    holdout_df = load_csv_rows(REPO_ROOT / "holdout" / "holdout_images.csv")

    shelf_downloaded, shelf_skipped, shelf_failed = download_records(
        images_df.to_dict(orient="records"),
        id_column="image_id",
        url_column="image_url",
        output_dir=IMAGES_DIR,
        label="shelf",
    )
    thumb_downloaded, thumb_skipped, thumb_failed = download_records(
        catalogue_df.to_dict(orient="records"),
        id_column="sku_id",
        url_column="thumbnail_url",
        output_dir=THUMBS_DIR,
        label="thumb",
    )
    holdout_downloaded, holdout_skipped, holdout_failed = download_records(
        holdout_df.to_dict(orient="records"),
        id_column="image_id",
        url_column="image_url",
        output_dir=HOLDOUT_DIR,
        label="holdout",
    )

    print("\nDownload summary")
    print(f"- Shelf images downloaded: {shelf_downloaded}")
    print(f"- Thumbnails downloaded: {thumb_downloaded}")
    print(f"- Holdout images downloaded: {holdout_downloaded}")
    print(f"- Files skipped: {shelf_skipped + thumb_skipped + holdout_skipped}")
    print(f"- Failed downloads: {shelf_failed + thumb_failed + holdout_failed}")


if __name__ == "__main__":
    main()
