# Neurolabs Product Engineer — Data Pack

Real, anonymized production data from our retail-shelf computer-vision platform (grocery / confectionery category).

**Licence — please read.** The images and data are provided **for this evaluation exercise only**: non-commercial, no redistribution, no model training beyond this challenge. We don't use submissions in production. By downloading, you agree to the full terms in **`LICENSE.txt`**.

Everything here is self-contained: image and thumbnail URLs point to this same public folder.

## Files
```
LICENSE.txt                evaluation-only, non-commercial licence — read first
starter.py                 run this first — loads, joins, draws boxes
requirements.txt           convenience deps (edit freely)
images.csv / images.json   100 shelf photos + per-image stats (incl. image_width/height)
ground_truth.csv           5,645 human-verified labels (your dev set)
catalogue.csv              319 product SKUs + a reference thumbnail for every one
images/                    img_000.jpg … img_099.jpg
thumbs/                    one reference thumbnail per SKU (sku_XXXX.jpg)
holdout/                   15 NEW shelves we kept the answers to — your scored test (see below)
```

Start with **`starter.py`** — it removes the plumbing (load, join to catalogue, draw boxes) so you can spend your time on the actual build.

### `images.csv` / `images.json` — 100 shelf photos
| Column | Meaning |
|---|---|
| `image_id` | Stable ID (`img_000…img_099`). Join key to `ground_truth.csv`. |
| `image_url` | Public URL of the shelf photo. |
| `scene_type` | Retail scene: Main aisle / Secondary display / Checkout / Chilled / Shelf. |
| `image_width`, `image_height` | Pixel dimensions (boxes are in this space). |
| `annotations_count` | Products the detector found on the shelf. |
| `n_gt_annotations` | Human-verified boxes available for this image. |
| `n_gt_identified` | Of those, how many carry a real product identity (vs gap/no-SKU). |
| `n_distinct_sku` | Distinct SKUs on the shelf. |

### `ground_truth.csv` — 5,645 human-verified labels
One row per detected box on the 100 shelves, with its **human-verified label**.

| Column | Meaning |
|---|---|
| `image_id`, `image_url` | Which shelf (join to `images.csv`). |
| `box_id` | Stable ID for this box. |
| `x, y, w, h` | Bounding box in pixels of `image_url`. |
| `gt_*` | Human label: `gt_sku_id` (→ `catalogue.csv`) + name/brand/flavour/size/pack/container. Blank = the box is a **gap / non-product / untracked item**. |
| `gt_has_sku` | `True` if the box is a real product (4,291 rows); `False` for gaps/no-SKU (1,354 rows). |

**What this is — and isn't.** Boxes were auto-detected and then **human-verified**, so labels only exist where something was detected — a product missed by detection has no box. So **score identity *given a box*, not detection recall**, and don't penalize yourself for extra detections in unlabelled regions. The catalogue spans the full confectionery range (Cadbury, Oreo, Toblerone, Belvita, Dairylea, Maynards…). The genuinely hard cases are *size/format* and *variant/flavour* — the same product shows up in near-identical singles, multipacks, and sizes.

### `catalogue.csv` — 319 SKUs
`sku_id, name, barcode, brand, flavour, size, containerType, packagingSize, height/width/depth, thumbnail_url, status`. `sku_id` joins to `gt_sku_id`. **Every SKU has a clean reference thumbnail** (`thumbnail_url` → `thumbs/`).

## Holdout — your scored test (`holdout/`)
These are **15 different shelves** (not in the 100) that we kept the answers to. This is how we measure accuracy fairly — the ground truth above is for you to develop against; the holdout is the exam.

- `holdout/holdout_to_predict.csv` — boxes with coordinates but **no identity** (722 boxes). Predict the catalogue `sku_id` for each `box_id`.
- `holdout/holdout_images.csv` — the 15 images + dimensions.
- `holdout/predictions_template.csv` — fill the `pred_sku_id` column and **send it back with your submission.**

`starter.py` writes a `predictions.csv` template for you. We score exact-SKU + brand/flavour/size against our held-back labels. Predicting from the dev labels won't help you here — these shelves aren't in it.

## How to approach recognition (no training required)
You do **not** need to train or fine-tune anything. Paths that work:
- **VLM as an attribute reader** — crop a box, ask a VLM for brand / flavour / size / format, then match to `catalogue.csv`. The attribute tuple is near-unique (292 of 319 SKUs resolve uniquely), so this alone gets you a real recogniser.
- **Embedding retrieval** — embed the `thumbs/` reference images and each box crop with an off-the-shelf encoder (CLIP / SigLIP / DINO) and take nearest-neighbour. *Size/format* and *variant* will be the hard part — that's the interesting bit.
- **Mix** — retrieve a shortlist, then use a VLM to disambiguate.

Pick one. We care how you drive the agent and the judgement you show — not which library you choose.

## Practical notes
- **Retrieval space is all 319 SKUs.** The dev labels only cover 184 of them — index against the whole catalogue so you're not blind to the rest.
- **Dev vs holdout:** self-measure on `ground_truth.csv` (you have those labels). We score your `holdout/predictions.csv` on exact-SKU + brand/flavour/size; **blank predictions count as wrong.**
- **Environment / cost:** `requirements.txt` is a convenience, edit freely. An embedding model means a one-off download; a VLM means ~700 calls for the holdout — batch and mind cost. Recent `transformers` (5.x) changed `get_image_features` to return an object, not a tensor — pin if it bites.

## Notes
- `packagingSize` encodes pack count (`1P/1` = single, `4P/2` = 4-pack); `size` is unit size/weight.
- **Real data is messy:** a few rows have `size`/`pack` values swapped, and some catalogue fields are sparse. Handling that cleanly is part of the test.
- Bounding boxes are pixel coordinates of the linked image.
- Not every detected product has ground truth — `ground_truth.csv` is the verified subset.
