"""ViT-B/16 authenticity head (authentic vs tampered).

Frozen ImageNet ViT embeddings → synthetic-trained logistic head.
This is NOT a calibrated authenticity estimate for final PASS/HIGH-RISK decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from app.ml.authenticity_head import (
    clear_head_cache,
    load_head,
    predict_proba,
    save_head,
    train_binary_logreg,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "vit"
WEIGHTS_PATH = MODEL_DIR / "authenticity_head.npz"
META_PATH = MODEL_DIR / "authenticity_head_meta.json"

LABELS = ("authentic", "tampered")
EMBED_DIM = 768


def _synthetic_dataset(
    samples_per_class: int = 120, seed: int = 19
) -> tuple[np.ndarray, np.ndarray]:
    """Authorized synthetic embedding proxy — not real government document images."""
    rng = np.random.default_rng(seed)
    auth_mean = rng.normal(0.0, 0.14, size=EMBED_DIM)
    auth_mean[:48] += 0.32
    tamp_mean = rng.normal(0.0, 0.14, size=EMBED_DIM)
    tamp_mean[48:96] += 0.5
    tamp_mean[150:170] -= 0.35

    xs, ys = [], []
    for _ in range(samples_per_class):
        a = auth_mean + rng.normal(0.0, 0.11, size=EMBED_DIM)
        xs.append(a / (np.linalg.norm(a) + 1e-12))
        ys.append(0)
        t = tamp_mean + rng.normal(0.0, 0.17, size=EMBED_DIM)
        spikes = rng.choice(EMBED_DIM, size=6, replace=False)
        t[spikes] += rng.uniform(0.35, 0.85, size=6)
        xs.append(t / (np.linalg.norm(t) + 1e-12))
        ys.append(1)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


def train_and_save_vit_head(samples_per_class: int = 120) -> Dict[str, Any]:
    x, y = _synthetic_dataset(samples_per_class=samples_per_class)
    w, b = train_binary_logreg(x, y)
    meta = {
        "model_name": "vit_b16_authenticity_logreg",
        "model_version": "sih-phase6-synthetic-v1",
        "backbone": "vit_b_16",
        "embedding_dim": EMBED_DIM,
        "classes": list(LABELS),
        "dataset": "synthetic_embedding_proxy_v1",
        "samples_per_class": samples_per_class,
        "calibrated": False,
        "note": (
            "Trained on synthetic embedding distributions only — "
            "not on real government identity documents. "
            "Probabilities are uncalibrated review signals, not legal authenticity."
        ),
    }
    save_head(WEIGHTS_PATH, META_PATH, w, b, meta)
    clear_head_cache()
    return meta


def ensure_vit_head() -> Dict[str, Any]:
    model = load_head(WEIGHTS_PATH, META_PATH)
    if model is not None:
        return model["meta"]
    return train_and_save_vit_head()


def score_vit_authenticity(embedding: np.ndarray) -> Dict[str, Any]:
    """Score a 768-d ViT embedding with the authenticity head."""
    try:
        ensure_vit_head()
        model = load_head(WEIGHTS_PATH, META_PATH)
        if model is None:
            return {
                "status": "NOT_ASSESSED",
                "message": "ViT authenticity head unavailable.",
                "calibrated": False,
                "concern": "NOT_SCORED",
            }
        vec = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if vec.shape[0] != EMBED_DIM:
            return {
                "status": "NOT_ASSESSED",
                "message": f"Embedding dim {vec.shape[0]} != {EMBED_DIM}.",
                "calibrated": False,
                "concern": "NOT_SCORED",
            }
        probs = predict_proba(vec, model["weights"], model["bias"])
        auth_p = float(probs[0])
        tamp_p = float(probs[1])
        meta = model["meta"]
        return {
            "status": "SCORED",
            "authentic_probability": round(auth_p, 4),
            "tampered_probability": round(tamp_p, 4),
            "predicted_label": LABELS[int(np.argmax(probs))],
            "model_version": meta.get("model_version", "unknown"),
            "dataset": meta.get("dataset"),
            "calibrated": False,
            "concern": "UNCALIBRATED_SYNTHETIC",
            "note": meta.get(
                "note",
                "Uncalibrated synthetic-trained head — not a final authenticity verdict.",
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "NOT_ASSESSED",
            "error": str(exc),
            "calibrated": False,
            "concern": "NOT_SCORED",
            "note": "ViT authenticity head failed. Not a fraud verdict.",
        }
