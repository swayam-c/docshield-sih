"""CLIP supporting evidence — document-type alignment, not authenticity.

CLIP text–image similarity must never be treated as proof a document is genuine.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from app.ml.device import use_mock_backends
from app.ml.summarize import mock_embedding

# SIH-oriented visual prompts (generic English — not government specs)
CLIP_PROTOTYPE_PROMPTS: Dict[str, str] = {
    "PASSPORT": "a scanned passport identity page with a photo and printed fields",
    "VISA": "a visa sticker or visa page with travel endorsement fields",
    "NATIONAL_ID": "a national identity card with a portrait photo",
    "DRIVING_LICENCE": "a driving licence or driver license card",
    "PERMIT": "a residence or work permit identity document",
    "OTHER": "a generic government or identity document form",
    "UNKNOWN": "an unclear or incomplete identity document image",
}

# Map fine subtypes → SIH category for alignment reporting
_SUBTYPE_TO_SIH = {
    "PAN": "NATIONAL_ID",
    "AADHAAR": "NATIONAL_ID",
}


def _l2(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2(a), _l2(b)))


def _mock_prototype_vectors(dim: int = 512) -> Dict[str, np.ndarray]:
    """Deterministic synthetic text-prototype stand-ins for mock ML mode."""
    out = {}
    for i, label in enumerate(CLIP_PROTOTYPE_PROMPTS):
        out[label] = mock_embedding(dim, seed=1000 + i * 17)
    return out


def _score_against_prototypes(
    image_vec: np.ndarray, prototypes: Dict[str, np.ndarray]
) -> Dict[str, float]:
    scores = {lab: round(_cosine(image_vec, proto), 4) for lab, proto in prototypes.items()}
    return scores


def _softmax_dict(scores: Dict[str, float], temperature: float = 0.15) -> Dict[str, float]:
    labs = list(scores.keys())
    arr = np.array([scores[l] for l in labs], dtype=np.float64)
    # shift from [-1,1]-ish cosine into positive logits
    z = arr / max(temperature, 1e-6)
    z = z - np.max(z)
    e = np.exp(z)
    e = e / (e.sum() + 1e-12)
    return {lab: round(float(e[i]), 4) for i, lab in enumerate(labs)}


def _encode_text_prototypes_real() -> Optional[Dict[str, np.ndarray]]:
    try:
        import open_clip
        import torch

        from app.ml.device import get_device
        from app.ml.registry import get_backbone

        model, _ = get_backbone("clip_vit_b32")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        device = get_device()
        prompts = list(CLIP_PROTOTYPE_PROMPTS.values())
        tokens = tokenizer(prompts).to(device)
        with torch.inference_mode():
            text_feats = model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            arr = text_feats.detach().float().cpu().numpy()
        return {
            lab: arr[i]
            for i, lab in enumerate(CLIP_PROTOTYPE_PROMPTS.keys())
        }
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger("docshield.clip").warning(
            "CLIP text prototype encode failed: %s", exc
        )
        return None


def build_clip_supporting_evidence(
    embedding: np.ndarray,
    *,
    predicted_document_type: Optional[str] = None,
    backend: str = "mock",
) -> Dict[str, Any]:
    """Build CLIP supporting-evidence payload from an image embedding."""
    vec = np.asarray(embedding, dtype=np.float64).reshape(-1)
    dim = int(vec.shape[0])

    if backend == "mock" or use_mock_backends():
        prototypes = _mock_prototype_vectors(dim=dim)
        method = "mock_prototype_cosine"
    else:
        prototypes = _encode_text_prototypes_real()
        if prototypes is None:
            # Fall back to mock prototypes rather than inventing authenticity
            prototypes = _mock_prototype_vectors(dim=dim)
            method = "fallback_mock_prototype_cosine"
        else:
            method = "open_clip_text_image_cosine"

    raw_scores = _score_against_prototypes(vec, prototypes)
    probs = _softmax_dict(raw_scores)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_prob = ranked[0]

    predicted = predicted_document_type or "UNKNOWN"
    sih_pred = _SUBTYPE_TO_SIH.get(predicted, predicted)
    if sih_pred not in probs:
        sih_pred = "OTHER" if "OTHER" in probs else top_label
    alignment_to_predicted = probs.get(sih_pred)

    agreement = "NOT_ASSESSED"
    if predicted and alignment_to_predicted is not None:
        if top_label == sih_pred and top_prob >= 0.2:
            agreement = "ALIGNED"
        elif alignment_to_predicted >= 0.15:
            agreement = "PARTIAL"
        else:
            agreement = "DIVERGENT"

    unit = _l2(vec)
    fingerprint = [round(float(x), 6) for x in unit[:16].tolist()]

    return {
        "status": "SUPPORTING_SCORED",
        "role": "supporting_evidence",
        "concern": "NOT_AUTHENTICITY",
        "method": method,
        "document_type_alignment": {
            "top_label": top_label,
            "top_probability": top_prob,
            "scores": probs,
            "raw_cosine": raw_scores,
            "classifier_label": predicted,
            "classifier_sih_category": sih_pred,
            "alignment_to_classifier": alignment_to_predicted,
            "agreement": agreement,
            "note": (
                "CLIP text–image similarity is a supporting cue for visual type "
                "alignment only. It is NOT proof of document authenticity or genuineness."
            ),
        },
        "fusion_ready": {
            "embedding_dim": dim,
            "normalized": True,
            "fingerprint": fingerprint,
            "prototype_probability_vector": [probs[k] for k in CLIP_PROTOTYPE_PROMPTS],
            "prototype_labels": list(CLIP_PROTOTYPE_PROMPTS.keys()),
            "note": "Prepared for SIH Phase 8 feature fusion — not a fraud score.",
        },
        "calibrated": False,
        "note": (
            "CLIP is integrated as supporting evidence only. "
            "Do not interpret CLIP scores as authenticity."
        ),
    }
