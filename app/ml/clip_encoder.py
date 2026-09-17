"""CLIP supporting image features (not a fraud detector)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from app.ml.clip_evidence import build_clip_supporting_evidence
from app.ml.device import get_device, use_mock_backends
from app.ml.registry import clip_available, get_backbone, pil_from_bytes
from app.ml.summarize import mock_embedding, summarize_embedding


@torch.inference_mode()
def extract_clip_features(
    image_bytes: bytes,
    *,
    predicted_document_type: Optional[str] = None,
) -> Dict[str, Any]:
    if use_mock_backends():
        vec = mock_embedding(512, seed=47 + len(image_bytes) % 83)
        summary = summarize_embedding(vec, model="clip_vit_b32_mock", role="supporting_features")
        summary["backend"] = "mock"
        summary["supporting_evidence"] = build_clip_supporting_evidence(
            vec, predicted_document_type=predicted_document_type, backend="mock"
        )
        summary["note"] = (
            "CLIP mock supporting features. "
            "Similarity/alignment is NOT document authenticity."
        )
        return summary

    if not clip_available():
        return {
            "status": "NOT_ASSESSED",
            "model": "clip_vit_b32",
            "role": "supporting_features",
            "concern": "NOT_SCORED",
            "error": "open_clip not installed",
            "supporting_evidence": {
                "status": "NOT_ASSESSED",
                "concern": "NOT_AUTHENTICITY",
                "note": "CLIP unavailable — supporting evidence not assessed.",
            },
            "note": "CLIP supporting features unavailable. Not a fraud verdict.",
        }

    try:
        model, preprocess = get_backbone("clip_vit_b32")
        img = pil_from_bytes(image_bytes)
        tensor = preprocess(img).unsqueeze(0).to(get_device())
        feats = model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vec = feats.squeeze(0).detach().float().cpu().numpy()
        summary = summarize_embedding(
            vec, model="clip_vit_b32", role="supporting_features"
        )
        summary["backend"] = "open_clip"
        summary["weights"] = "openai"
        summary["supporting_evidence"] = build_clip_supporting_evidence(
            vec, predicted_document_type=predicted_document_type, backend="open_clip"
        )
        summary["note"] = (
            "CLIP supporting features only. "
            "CLIP similarity does NOT prove a document is genuine."
        )
        return summary
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "NOT_ASSESSED",
            "model": "clip_vit_b32",
            "role": "supporting_features",
            "concern": "NOT_SCORED",
            "error": str(exc),
            "supporting_evidence": {
                "status": "NOT_ASSESSED",
                "concern": "NOT_AUTHENTICITY",
            },
            "note": "CLIP features unavailable. Not a fraud verdict.",
        }
