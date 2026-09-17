"""Feature fusion — transparent logistic head over multimodal signals.

Fuses EfficientNet / ViT / CLIP / OCR / quality / MRZ cues into an uncalibrated
authentic-vs-tampered score. Not a legal authenticity verdict; calibration is later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ml.authenticity_head import (
    clear_head_cache,
    load_head,
    predict_proba,
    save_head,
    train_binary_logreg,
)
from app.ml.clip_evidence import CLIP_PROTOTYPE_PROMPTS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "fusion"
WEIGHTS_PATH = MODEL_DIR / "fusion_logreg.npz"
META_PATH = MODEL_DIR / "fusion_logreg_meta.json"

ENET_FP = 12
VIT_FP = 12
CLIP_FP = 16
CLIP_PROTO = len(CLIP_PROTOTYPE_PROMPTS)  # 7


def _feature_names() -> List[str]:
    names: List[str] = []
    names += [f"enet_fp_{i}" for i in range(ENET_FP)]
    names += ["enet_auth", "enet_tamp", "enet_present"]
    names += [f"vit_fp_{i}" for i in range(VIT_FP)]
    names += ["vit_auth", "vit_tamp", "vit_present"]
    names += [f"clip_fp_{i}" for i in range(CLIP_FP)]
    names += [f"clip_proto_{i}" for i in range(CLIP_PROTO)]
    names += ["clip_present"]
    names += [
        "evidence_quality",
        "ocr_confidence",
        "field_completeness",
        "doc_type_confidence",
        "mrz_check_pass_rate",
        "mrz_parsed",
    ]
    return names


FEATURE_NAMES = tuple(_feature_names())
FUSION_DIM = len(FEATURE_NAMES)


def _pad_fp(fp: Optional[List[float]], n: int) -> List[float]:
    vals = [float(x) for x in (fp or [])[:n]]
    if len(vals) < n:
        vals += [0.0] * (n - len(vals))
    return vals


def _mrz_pass_rate(mrz: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    if not mrz:
        return 0.0, 0.0
    if mrz.get("status") != "PARSED":
        return 0.0, 0.0
    checklist = mrz.get("checklist") or []
    if not checklist:
        ok = 1.0 if mrz.get("check_digits_ok") else 0.0
        return ok, 1.0
    scored = [c for c in checklist if c.get("status") in {"PASS", "FAIL"}]
    if not scored:
        return 0.0, 1.0
    passes = sum(1 for c in scored if c.get("status") == "PASS")
    return passes / len(scored), 1.0


def build_fusion_vector(
    *,
    vision_features: Optional[Dict[str, Any]] = None,
    evidence_quality: float = 0.0,
    ocr_confidence: Optional[float] = None,
    field_completeness: Optional[float] = None,
    doc_type_confidence: float = 0.0,
    mrz: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Assemble normalized multimodal feature vector + availability metadata."""
    vision_features = vision_features or {}
    enet = vision_features.get("efficientnet") or {}
    vit = vision_features.get("vit") or {}
    clip = vision_features.get("clip") or {}
    enet_head = enet.get("authenticity_head") or {}
    vit_head = vit.get("authenticity_head") or {}
    clip_support = clip.get("supporting_evidence") or {}
    fusion_ready = clip_support.get("fusion_ready") or {}

    enet_present = 1.0 if enet.get("status") == "FEATURES_EXTRACTED" else 0.0
    vit_present = 1.0 if vit.get("status") == "FEATURES_EXTRACTED" else 0.0
    clip_present = 1.0 if clip.get("status") == "FEATURES_EXTRACTED" else 0.0

    parts: List[float] = []
    parts += _pad_fp(enet.get("fingerprint"), ENET_FP)
    parts += [
        float(enet_head.get("authentic_probability") or 0.0),
        float(enet_head.get("tampered_probability") or 0.0),
        enet_present,
    ]
    parts += _pad_fp(vit.get("fingerprint"), VIT_FP)
    parts += [
        float(vit_head.get("authentic_probability") or 0.0),
        float(vit_head.get("tampered_probability") or 0.0),
        vit_present,
    ]
    parts += _pad_fp(fusion_ready.get("fingerprint") or clip.get("fingerprint"), CLIP_FP)
    proto = fusion_ready.get("prototype_probability_vector") or [0.0] * CLIP_PROTO
    parts += _pad_fp(proto, CLIP_PROTO)
    parts.append(clip_present)

    mrz_rate, mrz_parsed = _mrz_pass_rate(mrz)
    parts += [
        float(evidence_quality or 0.0),
        float(ocr_confidence if ocr_confidence is not None else 0.0),
        float(field_completeness if field_completeness is not None else 0.0),
        float(doc_type_confidence or 0.0),
        float(mrz_rate),
        float(mrz_parsed),
    ]

    vec = np.asarray(parts, dtype=np.float64)
    assert vec.shape[0] == FUSION_DIM, f"fusion dim {vec.shape[0]} != {FUSION_DIM}"
    meta = {
        "feature_names": list(FEATURE_NAMES),
        "dim": FUSION_DIM,
        "modalities_present": {
            "efficientnet": bool(enet_present),
            "vit": bool(vit_present),
            "clip": bool(clip_present),
            "ocr": ocr_confidence is not None,
            "fields": field_completeness is not None,
            "mrz": bool(mrz_parsed),
        },
    }
    return vec, meta


def _synthetic_fusion_dataset(
    samples_per_class: int = 160, seed: int = 31
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(samples_per_class):
        # Authentic-leaning
        a = np.zeros(FUSION_DIM, dtype=np.float64)
        a[0:ENET_FP] = rng.normal(0.05, 0.08, ENET_FP)
        a[ENET_FP] = rng.uniform(0.55, 0.9)  # enet_auth
        a[ENET_FP + 1] = 1.0 - a[ENET_FP]
        a[ENET_FP + 2] = 1.0
        base = ENET_FP + 3
        a[base : base + VIT_FP] = rng.normal(0.04, 0.07, VIT_FP)
        a[base + VIT_FP] = rng.uniform(0.55, 0.88)
        a[base + VIT_FP + 1] = 1.0 - a[base + VIT_FP]
        a[base + VIT_FP + 2] = 1.0
        base = ENET_FP + 3 + VIT_FP + 3
        a[base : base + CLIP_FP] = rng.normal(0.03, 0.06, CLIP_FP)
        # passport-ish proto peak
        proto = rng.dirichlet(np.ones(CLIP_PROTO) * 0.7)
        proto[0] += 0.25
        proto = proto / proto.sum()
        a[base + CLIP_FP : base + CLIP_FP + CLIP_PROTO] = proto
        a[base + CLIP_FP + CLIP_PROTO] = 1.0
        tail = base + CLIP_FP + CLIP_PROTO + 1
        a[tail] = rng.uniform(0.6, 0.95)  # quality
        a[tail + 1] = rng.uniform(0.55, 0.95)
        a[tail + 2] = rng.uniform(0.4, 0.9)
        a[tail + 3] = rng.uniform(0.45, 0.9)
        a[tail + 4] = rng.uniform(0.5, 1.0)
        a[tail + 5] = float(rng.random() > 0.4)
        xs.append(a)
        ys.append(0)

        # Tampered-leaning
        t = np.zeros(FUSION_DIM, dtype=np.float64)
        t[0:ENET_FP] = rng.normal(-0.05, 0.12, ENET_FP)
        t[ENET_FP] = rng.uniform(0.1, 0.45)
        t[ENET_FP + 1] = 1.0 - t[ENET_FP]
        t[ENET_FP + 2] = 1.0
        base = ENET_FP + 3
        t[base : base + VIT_FP] = rng.normal(-0.04, 0.11, VIT_FP)
        t[base + VIT_FP] = rng.uniform(0.1, 0.42)
        t[base + VIT_FP + 1] = 1.0 - t[base + VIT_FP]
        t[base + VIT_FP + 2] = 1.0
        base = ENET_FP + 3 + VIT_FP + 3
        t[base : base + CLIP_FP] = rng.normal(-0.03, 0.1, CLIP_FP)
        proto = rng.dirichlet(np.ones(CLIP_PROTO))
        t[base + CLIP_FP : base + CLIP_FP + CLIP_PROTO] = proto
        t[base + CLIP_FP + CLIP_PROTO] = 1.0
        tail = base + CLIP_FP + CLIP_PROTO + 1
        t[tail] = rng.uniform(0.15, 0.55)
        t[tail + 1] = rng.uniform(0.1, 0.5)
        t[tail + 2] = rng.uniform(0.0, 0.4)
        t[tail + 3] = rng.uniform(0.15, 0.5)
        t[tail + 4] = rng.uniform(0.0, 0.5)
        t[tail + 5] = float(rng.random() > 0.6)
        xs.append(t)
        ys.append(1)

    return np.asarray(xs), np.asarray(ys, dtype=np.int64)


def train_and_save_fusion(samples_per_class: int = 160) -> Dict[str, Any]:
    x, y = _synthetic_fusion_dataset(samples_per_class=samples_per_class)
    w, b = train_binary_logreg(x, y, lr=0.2, epochs=450)
    meta = {
        "model_name": "multimodal_fusion_logreg",
        "model_version": "sih-phase8-synthetic-v1",
        "architecture": "logistic_regression",
        "embedding_dim": FUSION_DIM,
        "feature_names": list(FEATURE_NAMES),
        "classes": ["authentic", "tampered"],
        "dataset": "synthetic_multimodal_proxy_v1",
        "samples_per_class": samples_per_class,
        "calibrated": False,
        "note": (
            "Trained on synthetic multimodal feature distributions only — "
            "not on real government identity documents. "
            "Fusion output is uncalibrated and is NOT a final authenticity_estimate."
        ),
    }
    save_head(WEIGHTS_PATH, META_PATH, w, b, meta)
    clear_head_cache()
    return meta


def ensure_fusion_model() -> Dict[str, Any]:
    model = load_head(WEIGHTS_PATH, META_PATH)
    if model is not None:
        return model["meta"]
    return train_and_save_fusion()


def fuse_evidence(
    *,
    vision_features: Optional[Dict[str, Any]] = None,
    evidence_quality: float = 0.0,
    ocr_confidence: Optional[float] = None,
    field_completeness: Optional[float] = None,
    doc_type_confidence: float = 0.0,
    mrz: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run multimodal fusion. Returns uncalibrated authentic/tampered probabilities."""
    try:
        ensure_fusion_model()
        model = load_head(WEIGHTS_PATH, META_PATH)
        if model is None:
            return {
                "status": "NOT_ASSESSED",
                "message": "Fusion model unavailable.",
                "calibrated": False,
                "concern": "NOT_SCORED",
            }

        vec, vec_meta = build_fusion_vector(
            vision_features=vision_features,
            evidence_quality=evidence_quality,
            ocr_confidence=ocr_confidence,
            field_completeness=field_completeness,
            doc_type_confidence=doc_type_confidence,
            mrz=mrz,
        )
        # Require at least one vision backbone
        present = vec_meta["modalities_present"]
        if not (present["efficientnet"] or present["vit"] or present["clip"]):
            return {
                "status": "NOT_ASSESSED",
                "message": "No vision modalities available for fusion.",
                "calibrated": False,
                "concern": "NOT_SCORED",
                "vector": vec_meta,
            }

        probs = predict_proba(vec, model["weights"], model["bias"])
        auth_p = float(probs[0])
        tamp_p = float(probs[1])
        meta = model["meta"]

        # Head agreement signal (not authenticity)
        enet = ((vision_features or {}).get("efficientnet") or {}).get("authenticity_head") or {}
        vit = ((vision_features or {}).get("vit") or {}).get("authenticity_head") or {}
        head_auth = []
        if enet.get("status") == "SCORED":
            head_auth.append(float(enet.get("authentic_probability") or 0))
        if vit.get("status") == "SCORED":
            head_auth.append(float(vit.get("authentic_probability") or 0))
        mean_head = float(np.mean(head_auth)) if head_auth else None
        agreement = "NOT_ASSESSED"
        if mean_head is not None:
            if abs(mean_head - auth_p) < 0.15:
                agreement = "AGREE"
            elif abs(mean_head - auth_p) < 0.3:
                agreement = "PARTIAL"
            else:
                agreement = "DIVERGE"

        return {
            "status": "FUSED",
            "architecture": "logistic_regression",
            "authentic_probability": round(auth_p, 4),
            "tampered_probability": round(tamp_p, 4),
            "predicted_label": "authentic" if auth_p >= tamp_p else "tampered",
            "model_version": meta.get("model_version", "unknown"),
            "dataset": meta.get("dataset"),
            "calibrated": False,
            "concern": "UNCALIBRATED_SYNTHETIC",
            "head_agreement": agreement,
            "modalities_present": present,
            "feature_dim": FUSION_DIM,
            "vector_fingerprint": [round(float(x), 5) for x in vec[:12].tolist()],
            "note": meta.get(
                "note",
                "Uncalibrated fusion score — not authenticity_estimate / not legal authenticity.",
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "NOT_ASSESSED",
            "error": str(exc),
            "calibrated": False,
            "concern": "NOT_SCORED",
            "note": "Feature fusion failed. Not a fraud verdict.",
        }
