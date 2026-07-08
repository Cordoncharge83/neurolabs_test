#!/usr/bin/env python3
"""Evaluate the recognition baseline against ground-truth SKU boxes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.predict import load_prediction_context, predict_image

GROUND_TRUTH_PATH = REPO_ROOT / "ground_truth.csv"
CROPS_DIR = REPO_ROOT / "data" / "crops"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
RESULTS_PATH = RESULTS_DIR / "baseline_predictions.csv"


def parse_bool(value: Any) -> bool:
    """Parse a CSV boolean-like value into a Python bool."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False

    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """Load the ground-truth CSV rows."""
    return pd.read_csv(csv_path)


def evaluate_baseline() -> pd.DataFrame:
    """Evaluate the baseline prediction pipeline for valid ground-truth SKU rows."""
    ground_truth_df = load_ground_truth(GROUND_TRUTH_PATH)
    context = load_prediction_context()

    results: list[dict[str, Any]] = []
    skipped_invalid = 0
    skipped_missing_crop = 0
    correct = 0
    incorrect = 0

    for index, row in enumerate(ground_truth_df.to_dict(orient="records"), start=1):
        image_id = str(row.get("image_id", "")).strip()
        box_id = str(row.get("box_id", "")).strip()
        gt_sku_id = str(row.get("gt_sku_id", "")).strip()
        has_sku = parse_bool(row.get("gt_has_sku"))

        if not has_sku or not gt_sku_id:
            skipped_invalid += 1
            continue

        crop_path = CROPS_DIR / image_id / f"{box_id}.jpg"
        if not crop_path.exists():
            skipped_missing_crop += 1
            print(f"Warning: skipping {image_id}/{box_id}: crop not found at {crop_path}")
            results.append(
                {
                    "image_id": image_id,
                    "box_id": box_id,
                    "gt_sku_id": gt_sku_id,
                    "pred_sku_id": "",
                    "similarity": np.nan,
                    "correct": False,
                    "status": "skipped_missing_crop",
                }
            )
            continue

        try:
            pred_sku_id, similarity, _ = predict_image(context, crop_path)
        except (FileNotFoundError, ValueError) as error:
            skipped_missing_crop += 1
            print(f"Warning: failed to evaluate {image_id}/{box_id}: {error}")
            results.append(
                {
                    "image_id": image_id,
                    "box_id": box_id,
                    "gt_sku_id": gt_sku_id,
                    "pred_sku_id": "",
                    "similarity": np.nan,
                    "correct": False,
                    "status": "skipped_prediction_error",
                }
            )
            continue

        is_correct = str(pred_sku_id).strip() == gt_sku_id
        if is_correct:
            correct += 1
        else:
            incorrect += 1

        results.append(
            {
                "image_id": image_id,
                "box_id": box_id,
                "gt_sku_id": gt_sku_id,
                "pred_sku_id": pred_sku_id,
                "similarity": similarity,
                "correct": is_correct,
                "status": "evaluated",
            }
        )

        if index % 100 == 0:
            print(f"Progress: evaluated {index} crops")

    results_df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    total_evaluated = len(results_df[results_df["status"] == "evaluated"])
    accuracy = (correct / total_evaluated * 100.0) if total_evaluated else 0.0

    print("Baseline evaluation summary")
    print(f"- total evaluated: {total_evaluated}")
    print(f"- correct: {correct}")
    print(f"- incorrect: {incorrect}")
    print(f"- accuracy: {accuracy:.2f}%")
    print(f"- skipped invalid ground truth rows: {skipped_invalid}")
    print(f"- skipped missing crop files: {skipped_missing_crop}")

    return results_df


def main() -> None:
    """Run the recognition baseline evaluation and save detailed results."""
    evaluate_baseline()
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
