"""Extract EfficientNet + ViT + CLIP features for a document image."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.ml.clip_encoder import extract_clip_features
from app.ml.efficientnet import extract_efficientnet_features
from app.ml.vit import extract_vit_features


def extract_vision_features(
    image_bytes: bytes,
    *,
    predicted_document_type: Optional[str] = None,
) -> Dict[str, Any]:
    efficientnet = extract_efficientnet_features(image_bytes)
    vit = extract_vit_features(image_bytes)
    clip = extract_clip_features(
        image_bytes, predicted_document_type=predicted_document_type
    )

    statuses = [efficientnet.get("status"), vit.get("status"), clip.get("status")]
    if all(s == "FEATURES_EXTRACTED" for s in statuses):
        status = "OK"
    elif any(s == "FEATURES_EXTRACTED" for s in statuses):
        status = "PARTIAL"
    else:
        status = "NOT_ASSESSED"

    clip_support = (clip or {}).get("supporting_evidence") or {}

    return {
        "status": status,
        "efficientnet": efficientnet,
        "vit": vit,
        "clip": clip,
        "clip_supporting_evidence": clip_support,
        "fusion": {
            "status": "DEFERRED_TO_PIPELINE",
            "message": (
                "Full multimodal fusion (OCR/quality/MRZ + vision) runs in /api/analyze. "
                "This endpoint only reports backbone readiness."
            ),
            "inputs_ready": {
                "efficientnet": efficientnet.get("status") == "FEATURES_EXTRACTED",
                "vit": vit.get("status") == "FEATURES_EXTRACTED",
                "clip": clip.get("status") == "FEATURES_EXTRACTED",
                "clip_prototype_vector": bool(
                    (clip_support.get("fusion_ready") or {}).get("prototype_probability_vector")
                ),
            },
        },
        "note": (
            "EfficientNet and ViT may include uncalibrated synthetic authenticity heads. "
            "CLIP provides supporting type-alignment evidence only — not authenticity. "
            "Multimodal fusion runs in the analyze pipeline; calibrated authenticity_estimate is later."
        ),
    }
