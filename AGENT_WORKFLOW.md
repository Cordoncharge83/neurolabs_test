# AGENT_WORKFLOW.md

## AI Stack

I used two AI coding assistants throughout the project:

- GitHub Copilot (primary implementation agent)
- ChatGPT (architecture, review, prompt refinement, and technical discussions)

Copilot was responsible for implementing most of the code. ChatGPT was used as a design and review partner to validate architectural decisions, discuss alternatives, improve prompts sent to Copilot, and explain unfamiliar concepts when needed.

---

## Agent Setup

Before starting implementation, I created a repository-wide AI instructions file (`codes.instructions.md`) to establish consistent behavior.

The main rules included:

- implement one logical task at a time
- never execute scripts automatically
- explain changes after implementation
- keep modules separated
- avoid unnecessary abstractions
- prefer readable code over clever solutions

This significantly reduced unwanted agent behavior and kept changes incremental and reviewable.

---

## Delegation Strategy

### Delegated to the coding agent

- data download script
- crop extraction
- catalogue embedding generation
- prediction pipeline
- evaluation scripts
- holdout prediction generation
- FastAPI backend
- React dashboard implementation
- UI refinements
- refactoring

### Reviewed manually

Every generated change was reviewed before execution.

I manually:

- ran every script
- inspected outputs
- reviewed code for correctness
- fixed or requested changes whenever implementation differed from the intended architecture

No generated code was accepted without review.

---

## Examples where human judgement changed the outcome

### 1. Removing the vector database

My initial instinct was to use a vector database because of previous experience with multimodal retrieval systems.

After considering the actual scale (319 catalogue products), I decided that brute-force cosine similarity was simpler, easier to maintain, and effectively instantaneous.

This removed unnecessary infrastructure without affecting performance.

---

### 2. Model selection

I evaluated several embedding models.

| Model             | Accuracy |
| ----------------- | -------- |
| OpenCLIP ViT-B-32 | 37.17%   |
| OpenCLIP ViT-B-16 | 39.71%   |
| OpenCLIP ViT-L-14 | 44.95%   |
| DINOv2            | 7.76%    |

Although ViT-L-14 produced the best accuracy, inference time increased significantly.

I selected ViT-B-16 because it provided the best balance between recognition quality and responsiveness for an interactive product.

---

### 3. Correcting AI-generated evaluation logic

The initial evaluation included bounding boxes that intentionally had no associated SKU.

During review I noticed that `ground_truth.csv` contains rows where `gt_has_sku` is false.

I updated the evaluation pipeline to ignore those rows, preventing incorrect accuracy calculations.

This reinforced my workflow of reviewing every AI-generated implementation rather than assuming correctness.

---

## Overall workflow

My workflow followed a simple loop:

1. Define one small objective.
2. Ask the coding agent to implement it.
3. Review the generated code.
4. Execute and validate manually.
5. Measure results.
6. Decide the next step based on evidence.

Rather than asking the agent to build the entire application at once, I kept development incremental so that every step could be verified before moving forward.
