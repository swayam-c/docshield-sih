"""EfficientNet-B0 authenticity head (authentic vs tampered).

Frozen ImageNet backbone embeddings → synthetic-trained logistic head.
This is NOT a calibrated authenticity estimate for final PASS/HIGH-RISK decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.ml.authenticity_head import (
    clear_head_cache,
    load_head,
    predict_proba,
    save_head,
    train_binary_logreg,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "efficientnet"
WEIGHTS_PATH = MODEL_DIR / "authenticity_head.npz"
META_PATH = MODEL_DIR / "authenticity_head_meta.json"

LABELS = ("authentic", "tampered")
EMBED_DIM = 1280


def _synthetic_dataset(
    samples_per_class: int = 120, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Authorized synthetic embedding proxy — not real government document images."""
    rng = np.random.default_rng(seed)
    # Authentic: moderate energy, smooth spectrum
    auth_mean = rng.normal(0.0, 0.15, size=EMBED_DIM)
    auth_mean[:64] += 0.35
    # Tampered: shifted high-frequency / outlier dimensions
    tamp_mean = rng.normal(0.0, 0.15, size=EMBED_DIM)
    tamp_mean[64:128] += 0.55
    tamp_mean[200:220] -= 0.4

    xs, ys = [], []
    for _ in range(samples_per_class):
        a = auth_mean + rng.normal(0.0, 0.12, size=EMBED_DIM)
        xs.append(a / (np.linalg.norm(a) + 1e-12))
        ys.append(0)
        t = tamp_mean + rng.normal(0.0, 0.18, size=EMBED_DIM)
        # occasional spike artifacts
        spikes = rng.choice(EMBED_DIM, size=8, replace=False)
        t[spikes] += rng.uniform(0.4, 0.9, size=8)
        xs.append(t / (np.linalg.norm(t) + 1e-12))
        ys.append(1)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


def train_and_save_efficientnet_head(samples_per_class: int = 120) -> Dict[str, Any]:
    x, y = _synthetic_dataset(samples_per_class=samples_per_class)
    w, b = train_binary_logreg(x, y)
    meta = {
        "model_name": "efficientnet_b0_authenticity_logreg",
        "model_version": "sih-phase5-synthetic-v1",
        "backbone": "efficientnet_b0",
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


def ensure_efficientnet_head() -> Dict[str, Any]:
    model = load_head(WEIGHTS_PATH, META_PATH)
    if model is not None:
        return model["meta"]
    return train_and_save_efficientnet_head()


def score_efficientnet_authenticity(embedding: np.ndarray) -> Dict[str, Any]:
    """Score a 1280-d EfficientNet embedding with the authenticity head."""
    try:
        ensure_efficientnet_head()
        model = load_head(WEIGHTS_PATH, META_PATH)
        if model is None:
            return {
                "status": "NOT_ASSESSED",
                "message": "EfficientNet authenticity head unavailable.",
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
            "note": "EfficientNet authenticity head failed. Not a fraud verdict.",
        }
