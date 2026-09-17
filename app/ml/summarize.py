"""Feature summary helpers — store compact fingerprints, not full huge vectors in DB."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def summarize_embedding(vec: np.ndarray, *, model: str, role: str) -> Dict[str, Any]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr)) + 1e-12
    unit = arr / norm
    fingerprint = [round(float(x), 6) for x in unit[:12].tolist()]
    return {
        "status": "FEATURES_EXTRACTED",
        "model": model,
        "role": role,
        "embedding_dim": int(arr.shape[0]),
        "embedding_norm": round(norm, 6),
        "embedding_mean": round(float(arr.mean()), 6),
        "embedding_std": round(float(arr.std()), 6),
        "fingerprint": fingerprint,
        "concern": "NOT_SCORED",
        "note": (
            "Pretrained backbone features only (FEATURES_EXTRACTED). "
            "This is NOT a fraud/authenticity score. "
            "Dedicated authenticity heads (when present) are reported separately "
            "and remain UNCALIBRATED_SYNTHETIC until calibration."
        ),
    }


def mock_embedding(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)
