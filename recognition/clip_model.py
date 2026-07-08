#!/usr/bin/env python3
"""Minimal CLIP model loading helper for image embeddings."""

from __future__ import annotations

from typing import Any

import torch
import open_clip
from PIL import Image


def load_clip_model(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> tuple[Any, Any, Any]:
    """Load a pretrained CLIP model and preprocessing transform."""
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    return model, preprocess, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_image(model: Any, preprocess: Any, device: torch.device, image: Image.Image) -> torch.Tensor:
    """Encode a PIL image to a normalized CLIP embedding tensor."""
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()
