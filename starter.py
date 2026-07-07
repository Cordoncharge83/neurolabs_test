"""
Neurolabs Product Engineer — starter script.

Removes the boring plumbing so you can spend your time on the interesting part.
Shows: load the pack, join the ground-truth labels to the catalogue, draw the
boxes on a shelf, look at the label distribution, and produce a holdout submission.

Run from inside the unzipped pack folder (or it reads from the public URLs).
    python starter.py
Deps: pip install pillow requests
"""
import csv, io, os, urllib.request
from collections import Counter

BASE = "https://storage.googleapis.com/zia-web-public/jobs/ai-engineer-june-2026"

def load(name):
    """Load a CSV from the local folder if present, else from the public bucket."""
    if os.path.exists(name):
        return list(csv.DictReader(open(name)))
    return list(csv.DictReader(io.StringIO(urllib.request.urlopen(f"{BASE}/{name}").read().decode())))

def main():
    images    = load("images.csv")
    gt        = load("ground_truth.csv")
    catalogue = load("catalogue.csv")
    cat_by_id = {c["sku_id"]: c for c in catalogue}
    print(f"{len(images)} shelves | {len(gt)} labelled boxes | {len(catalogue)} SKUs")

    # --- 1. the labels: each box is either a product (gt_sku_id set) or a gap ---
    identified = [r for r in gt if r["gt_has_sku"] == "True"]
    gaps = len(gt) - len(identified)
    print(f"\nlabels: {len(identified)} identified products, {gaps} gaps/non-products")
    print("Build a recogniser, then score IT against gt_sku_id on the identified boxes")
    print("(score identity GIVEN A BOX; the gaps aren't products, and boxes only exist")
    print(" where something was detected — so this isn't a detection/recall benchmark).")

    # --- 2. join a label to the catalogue ---
    row = identified[0]
    sku = cat_by_id[row["gt_sku_id"]]
    print(f"\nexample: box {row['box_id']} -> {sku['name']} ({sku['brand']}, {sku['size']})")

    # the hard part, so you know where to spend effort:
    print("size mix:", dict(Counter(r["gt_size"] for r in identified if r["gt_size"]).most_common(5)))

    # --- 3. draw the boxes on one shelf (sanity-check your coordinate handling) ---
    try:
        from PIL import Image, ImageDraw
        img = images[0]
        local = f"images/{img['image_id']}.jpg"
        src = local if os.path.exists(local) else io.BytesIO(urllib.request.urlopen(img["image_url"]).read())
        im = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(im)
        for r in (r for r in gt if r["image_id"] == img["image_id"]):
            x, y, w, h = (float(r[k]) for k in "xywh")
            colour = (255, 103, 0) if r["gt_has_sku"] == "True" else (150, 150, 150)  # product vs gap
            d.rectangle([x, y, x + w, y + h], outline=colour, width=6)
        im.save("starter_overlay.jpg", quality=85)
        print(f"\nwrote starter_overlay.jpg for {img['image_id']} — boxes should sit on the products")
    except ImportError:
        print("\n(install pillow to render the overlay)")

    # --- 4. holdout: produce the submission you send us ---
    # holdout/ has 15 shelves + boxes with NO identity. Predict a catalogue sku_id per box.
    try:
        todo = load("holdout/holdout_to_predict.csv")
        with open("predictions.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["box_id", "pred_sku_id"])
            for r in todo:
                w.writerow([r["box_id"], ""])   # TODO: your recogniser predicts the sku_id here
        print(f"\nwrote predictions.csv template ({len(todo)} boxes). Fill pred_sku_id and send it with your submission.")
    except Exception:
        print("\n(holdout files load once the pack is fully published)")

if __name__ == "__main__":
    main()
