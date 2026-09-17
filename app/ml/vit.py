"""ViT-B/16 global feature extraction + optional authenticity head."""

from __future__ import annotations

from typing import Any, Dict

from app.ml.device import use_mock_backends
from app.ml.registry import forward_image, get_backbone, pil_from_bytes
from app.ml.summarize import mock_embedding, summarize_embedding
from app.ml.vit_head import score_vit_authenticity


def extract_vit_features(image_bytes: bytes) -> Dict[str, Any]:
    if use_mock_backends():
        vec = mock_embedding(768, seed=29 + len(image_bytes) % 89)
        summary = summarize_embedding(vec, model="vit_b16_mock", role="global_features")
        summary["backend"] = "mock"
        summary["authenticity_head"] = score_vit_authenticity(vec)
        return summary

    try:
        model, transform = get_backbone("vit_b16")
        img = pil_from_bytes(image_bytes)
        tensor = transform(img)
        feats = forward_image(model, tensor)
        vec = feats.numpy()
        summary = summarize_embedding(vec, model="vit_b_16", role="global_features")
        summary["backend"] = "torchvision"
        summary["weights"] = "ImageNet-1K"
        summary["authenticity_head"] = score_vit_authenticity(vec)
        return summary
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "NOT_ASSESSED",
            "model": "vit_b_16",
            "role": "global_features",
            "concern": "NOT_SCORED",
            "error": str(exc),
            "authenticity_head": {
                "status": "NOT_ASSESSED",
                "calibrated": False,
                "concern": "NOT_SCORED",
            },
            "note": "ViT features unavailable. Not a fraud verdict.",
        }
