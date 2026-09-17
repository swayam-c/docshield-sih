"""EfficientNet-B0 local feature extraction + optional authenticity head."""

from __future__ import annotations

from typing import Any, Dict

from app.ml.device import use_mock_backends
from app.ml.efficientnet_head import score_efficientnet_authenticity
from app.ml.registry import forward_image, get_backbone, pil_from_bytes
from app.ml.summarize import mock_embedding, summarize_embedding


def extract_efficientnet_features(image_bytes: bytes) -> Dict[str, Any]:
    if use_mock_backends():
        vec = mock_embedding(1280, seed=11 + len(image_bytes) % 97)
        summary = summarize_embedding(vec, model="efficientnet_b0_mock", role="local_features")
        summary["backend"] = "mock"
        summary["authenticity_head"] = score_efficientnet_authenticity(vec)
        return summary

    try:
        model, transform = get_backbone("efficientnet_b0")
        img = pil_from_bytes(image_bytes)
        tensor = transform(img)
        feats = forward_image(model, tensor)
        vec = feats.numpy()
        summary = summarize_embedding(vec, model="efficientnet_b0", role="local_features")
        summary["backend"] = "torchvision"
        summary["weights"] = "ImageNet-1K"
        summary["authenticity_head"] = score_efficientnet_authenticity(vec)
        return summary
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "NOT_ASSESSED",
            "model": "efficientnet_b0",
            "role": "local_features",
            "concern": "NOT_SCORED",
            "error": str(exc),
            "authenticity_head": {
                "status": "NOT_ASSESSED",
                "calibrated": False,
                "concern": "NOT_SCORED",
            },
            "note": "EfficientNet features unavailable. Not a fraud verdict.",
        }
