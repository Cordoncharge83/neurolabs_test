#!/usr/bin/env python3
"""Minimal CLIP model loading helper for image embeddings."""

from __future__ import annotations

import logging
import os
from typing import Any

import open_clip
import torch
from PIL import Image


logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _get_clip_config() -> tuple[str, str]:
    """Read the CLIP model configuration from environment variables with sensible defaults."""
    model_name = os.getenv("CLIP_MODEL_NAME", "ViT-B-16")
    pretrained = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b88k")
    return model_name, pretrained


def load_clip_model(model_name: str | None = None, pretrained: str | None = None) -> tuple[Any, Any, Any]:
    """Load a pretrained CLIP model and preprocessing transform."""
    if model_name is None or pretrained is None:
        configured_model_name, configured_pretrained = _get_clip_config()
        model_name = configured_model_name if model_name is None else model_name
        pretrained = configured_pretrained if pretrained is None else pretrained

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, preprocess, device


def encode_image(model: Any, preprocess: Any, device: torch.device, image: Image.Image) -> torch.Tensor:
    """Encode a PIL image to a normalized CLIP embedding tensor."""
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()
