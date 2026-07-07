# Neurolabs Product Engineer — Take-Home Task

**Apply by:** Thursday 17 July, 09:00 BST. Work at any pace until then.
**Process:** rolling review — early applications reviewed first → 2 offers shortly after.
**Location:** London (3 days/week) preferred · Edinburgh considered · remote for exceptional candidates.
**Submit to:** recruitment@neurolabs.ai
**Comp:** Competitive and set to your level. State your salary expectation in your submission.
**Time:** No fixed time box, but **aim for roughly half a day** — we reward judgement and agent-leverage, not hours logged. A focused, honest slice beats a sprawling one.

## Who we are
Neurolabs runs a computer-vision platform for retail execution: a shopper or auditor photographs a store shelf, and our pipeline detects every product, identifies the exact SKU against a catalogue, and turns that into insights (availability, share-of-shelf, pricing compliance). The pipeline spans capture apps, a vision/ML stack (detection → embedding retrieval → re-ranking), backend services, and dashboards.

## What we're actually testing
We are an AI-native team: **agents write the code, humans architect and review.** This task is deliberately *broader than you can hand-write in the time available.* We are not impressed by heroics at the keyboard — we are looking for someone who can **orchestrate AI agents (we use Claude / Claude Code) to produce more, faster, and still keep quality high.**

So: use agents aggressively. Show us how you do it.

---

## The data pack
You're given real, anonymized production data from our retail-shelf recognition workload (grocery / confectionery):

| File | What it is |
|---|---|
| `images.csv` / `images.json` | **100 real shelf photos** (public URLs), spread across real retail scene types (main aisle, secondary display, checkout…). Each row carries detection counts + SKU variety. |
| `ground_truth.csv` | **5,645 human-verified labels** over those 100 shelves (`gt_sku_id`, attributes, `gt_has_sku`). Boxes were auto-detected then human-verified, so score identity-*given-a-box*, not detection recall. See `DATA_README.md`. |
| `catalogue.csv` | **319 catalogue SKUs** (Cadbury, Oreo, Toblerone, Belvita, Dairylea…) — name, barcode, brand, flavour, size, packagingSize, containerType, dimensions, thumbnail. Every SKU has a reference thumbnail. |
| `holdout/` | **15 fresh shelves we kept the answers to** — 722 boxes with coordinates but no identity. Predict the SKU per box and submit `predictions.csv`; we score it. This is your honest accuracy check. |

Columns are documented in `DATA_README.md`. Run **`starter.py`** first — it does the loading, joining, box-drawing and holdout-template for you.

**The hard part:** telling apart near-identical variants — the same product in different sizes/formats, singles vs multipacks, and look-alike flavours. Handling the data's messiness (a few `size`/`pack` field swaps) is part of the test.

---

## Your task
Build a **thin but complete vertical slice** of a shelf-intelligence feature. It must span the stack:

1. **Ingest** — accept a shelf image (from the pack, or an uploaded one).
2. **Recognise** — detect and/or identify the products in it. Run your own model, call a VLM, use the catalogue thumbnails as references, or any mix — **no training required** (see "How to approach recognition" in `DATA_README.md`). Your call.
3. **Serve** — expose the result through a small API.
4. **Surface an insight** — a minimal dashboard/UI that shows the shelf and **one insight you chose**: e.g. share-of-shelf by brand, out-of-stock vs. planogram, pack/format mix, brand-block compliance — pick something you'd defend to a customer.
5. **Measure yourself** — score your recogniser against `ground_truth.csv` (identity given a box), then run it on the `holdout/` shelves and submit `predictions.csv`. **We care more about an honest, well-instrumented number than a high one.**

It's open-ended on purpose. Choose your own framing, models, and insight. A focused, working, honestly-measured slice beats a sprawling half-built one.

### Optional bonus track (explicitly scored)
**Synthetic data.** One genuinely hard problem in our world is telling apart near-identical variants — the same product in different sizes/formats, or singles vs multipacks. If you want to go further: generate or augment synthetic training/eval data (render, composite, or augment product views) and show it measurably helping — or at least stress-testing — your recognition step. This is where we find out if you can drive an agent into unfamiliar territory and come back with something real.

---

## What to submit (graded hardest first)
1. **A Loom (≤10 min)** of you *driving Claude / your agent* through a real part of this build. Show your prompting, how you break work down, how you catch and recover when the agent is wrong. This is the single most important artifact.
2. **A one-page workflow writeup** (`AGENT_WORKFLOW.md`): how you set up your agent(s) — CLAUDE.md, subagents/skills, prompt patterns — what you delegated vs. reviewed yourself, and the 2–3 moments your human judgement changed the outcome.
3. **The repo + a 1-paragraph README** on how to run it, and your honest accuracy number with how you computed it.
4. **`predictions.csv`** — your recogniser's SKU predictions for the `holdout/` shelves.

Keep it real: we want to see the *actual* mess of agent-driven development, not a polished story. A submission that honestly shows "the agent got this wrong and here's how I caught it" scores higher than a flawless-looking one with no insight into how it was built.

## Ground rules
- Use any models, frameworks, languages, and AI tools you like.
- Don't sink time into infra/polish; a rough but working slice is the goal.
- If you make an assumption, state it. Open-ended is not a trap — it's the test.

Questions & submissions: **recruitment@neurolabs.ai**.
