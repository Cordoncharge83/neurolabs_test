# Neurolabs Product Engineer Challenge

This repository contains a complete demo implementation of a shelf recognition workflow for the Neurolabs product engineering challenge.

## What is included

The project now covers the full end-to-end flow:

- downloading shelf images and thumbnails
- extracting product crops from shelf images
- embedding catalogue thumbnails with CLIP-style features
- predicting the best-matching SKU for each crop
- evaluating recognition quality against the provided ground truth
- generating holdout predictions
- exposing a small FastAPI backend for demo inference and analysis
- serving a React/Vite dashboard for browsing shelf images and reviewing predictions

## Project structure

- [scripts](scripts) — data download and crop extraction utilities
- [recognition](recognition) — CLIP model loading, catalogue embedding, and prediction logic
- [evaluation](evaluation) — baseline evaluation and holdout prediction generation
- [backend](backend) — FastAPI API for image listing, single-image prediction, and shelf analysis
- [frontend](frontend) — Vite + React dashboard for the demo experience

## Setup

### Python environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Frontend dependencies

```bash
cd frontend
npm install
```

## Data workflow

The repository includes scripts for preparing the local data assets:

```bash
python scripts/download_data.py
python scripts/extract_crops.py
python -m recognition.embed_catalogue
```

## Recognition and evaluation

Run the baseline evaluation and generate holdout predictions with:

```bash
python -m evaluation.evaluate_recognition
python -m evaluation.generate_predictions
```

## Backend

Start the API server:

```bash
python -m uvicorn backend.main:app --reload
```

The backend exposes:

- GET /health
- GET /images
- POST /predict
- POST /analyze

## Frontend

Start the dashboard locally:

```bash
cd frontend
npm run dev
```

The UI connects to the backend at http://localhost:8000 and lets you browse demo shelf images, run analysis, and inspect the recognition summary and product predictions.

## Notes

- The recognition stack uses CLIP-based embeddings and depends on PyTorch plus open-clip-torch.
- The frontend is intentionally lightweight and designed for demo purposes rather than production-scale UX.
