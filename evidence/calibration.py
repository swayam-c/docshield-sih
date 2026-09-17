"""SIH Phase 15 — probability calibration (Platt scaling on fusion scores).

Maps uncalibrated fusion authentic_probability → proxy-calibrated score used as
`authenticity_estimate` when evidence quality is sufficient.

Honesty:
- Fitted only on synthetic multimodal proxy scores (same family as fusion training).
- NOT trained on real government identity documents.
- NOT a legal authenticity verdict / NOT production reliability.
- Decision engine (PASS / REVIEW / HIGH-RISK) arrives in a later phase.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.ml.authenticity_head import clear_head_cache, load_head, save_head, train_binary_logreg
from evidence.fusion import (
    FEATURE_NAMES,
    FUSION_DIM,
    WEIGHTS_PATH as FUSION_WEIGHTS,
    META_PATH as FUSION_META,
    _synthetic_fusion_dataset,
    ensure_fusion_model,
)
from app.ml.authenticity_head import predict_proba as fusion_predict_proba

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "calibration"
WEIGHTS_PATH = MODEL_DIR / "platt_scaler.npz"
META_PATH = MODEL_DIR / "platt_scaler_meta.json"

EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(float(p), EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _raw_fusion_auth_probs(x: np.ndarray, fusion_model: Dict[str, Any]) -> np.ndarray:
    """Score each fusion vector with the frozen fusion head (class 0 = authentic)."""
    w = fusion_model["weights"]
    b = fusion_model["bias"]
    out = []
    for row in x:
        probs = fusion_predict_proba(row, w, b)
        out.append(float(probs[0]))
    return np.asarray(out, dtype=np.float64)


def train_and_save_calibration(samples_per_class: int = 200) -> Dict[str, Any]:
    """Fit 1-D Platt logistic on logit(raw_fusion_auth) using synthetic labels."""
    ensure_fusion_model()
    fusion_model = load_head(FUSION_WEIGHTS, FUSION_META)
    if fusion_model is None:
        raise RuntimeError("Fusion model required before calibration training.")

    x, y = _synthetic_fusion_dataset(samples_per_class=samples_per_class, seed=77)
    # y: 0=authentic, 1=tampered → for Platt we predict P(authentic)=1-y conceptually
    # Use authentic label: y_auth = 1 - y
    y_auth = (1 - y).astype(np.int64)
    raw = _raw_fusion_auth_probs(x, fusion_model)
    logits = np.asarray([[_logit(p)] for p in raw], dtype=np.float64)

    w, b = train_binary_logreg(logits, y_auth, lr=0.35, epochs=500)
    # For binary softmax head: P(auth) from class-0 vs class-1 on 1-D logit feature
    meta = {
        "model_name": "fusion_platt_scaler",
        "model_version": "sih-phase15-synthetic-platt-v1",
        "method": "platt_scaling",
        "input": "logit(fusion_authentic_probability)",
        "classes": ["authentic", "tampered"],
        "dataset": "synthetic_multimodal_proxy_v1",
        "samples_per_class": samples_per_class,
        "fusion_model_version": (fusion_model.get("meta") or {}).get("model_version"),
        "feature_dim_reference": FUSION_DIM,
        "feature_names_reference": list(FEATURE_NAMES),
        "proxy_calibrated": True,
        "production_calibrated": False,
        "note": (
            "Platt scaler fitted on synthetic fusion-score distributions only. "
            "proxy_calibrated=true means temperature/bias mapping was applied — "
            "NOT that the system is production-calibrated on real identity documents."
        ),
    }
    save_head(WEIGHTS_PATH, META_PATH, w, b, meta)
    clear_head_cache()
    return meta


def ensure_calibration_model() -> Dict[str, Any]:
    model = load_head(WEIGHTS_PATH, META_PATH)
    if model is not None:
        return model["meta"]
    return train_and_save_calibration()


def apply_platt(raw_authentic_probability: float, model: Dict[str, Any]) -> float:
    z = _logit(raw_authentic_probability)
    probs = fusion_predict_proba(np.asarray([z], dtype=np.float64), model["weights"], model["bias"])
    return float(probs[0])


def calibrate_fusion_score(
    *,
    fusion: Optional[Dict[str, Any]] = None,
    evidence_quality_ok: bool = True,
    decision: str = "PENDING",
) -> Dict[str, Any]:
    """Calibrate fusion authentic probability → authenticity_estimate candidate."""
    fusion = fusion or {}

    if decision == "INCONCLUSIVE" or not evidence_quality_ok:
        return {
            "status": "SKIPPED",
            "method": "platt_scaling",
            "raw_authentic_probability": fusion.get("authentic_probability"),
            "calibrated_authentic_probability": None,
            "authenticity_estimate": None,
            "proxy_calibrated": False,
            "production_calibrated": False,
            "concern": "CALIBRATION_SKIPPED_INSUFFICIENT_EVIDENCE",
            "note": (
                "Calibration skipped because the case is INCONCLUSIVE or evidence "
                "quality is insufficient. authenticity_estimate remains null."
            ),
        }

    if fusion.get("status") != "FUSED" or fusion.get("authentic_probability") is None:
        return {
            "status": "NOT_ASSESSED",
            "method": "platt_scaling",
            "raw_authentic_probability": fusion.get("authentic_probability"),
            "calibrated_authentic_probability": None,
            "authenticity_estimate": None,
            "proxy_calibrated": False,
            "production_calibrated": False,
            "concern": "FUSION_UNAVAILABLE",
            "note": "Fusion score unavailable — cannot calibrate.",
        }

    try:
        ensure_calibration_model()
        model = load_head(WEIGHTS_PATH, META_PATH)
        if model is None:
            return {
                "status": "ERROR",
                "method": "platt_scaling",
                "authenticity_estimate": None,
                "proxy_calibrated": False,
                "production_calibrated": False,
                "concern": "CALIBRATION_MODEL_MISSING",
                "note": "Calibration weights missing.",
            }

        raw = float(fusion["authentic_probability"])
        cal = round(apply_platt(raw, model), 4)
        meta = model["meta"]
        return {
            "status": "CALIBRATED",
            "method": "platt_scaling",
            "model_version": meta.get("model_version"),
            "raw_authentic_probability": round(raw, 4),
            "calibrated_authentic_probability": cal,
            "authenticity_estimate": cal,
            "proxy_calibrated": True,
            "production_calibrated": False,
            "calibrated": True,  # Platt applied (proxy); see production_calibrated
            "concern": "SYNTHETIC_PROXY_CALIBRATED",
            "note": meta.get("note")
            or (
                "Proxy-calibrated fusion score. Not a legal authenticity verdict "
                "and not production-calibrated on real government documents."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "method": "platt_scaling",
            "authenticity_estimate": None,
            "proxy_calibrated": False,
            "production_calibrated": False,
            "error": repr(exc),
            "concern": "CALIBRATION_ERROR",
            "note": "Calibration failed; authenticity_estimate left null.",
        }
