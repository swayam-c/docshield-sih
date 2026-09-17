"""Shared authenticity classification head utilities (logistic on frozen embeddings).

Trained only on authorized synthetic embedding proxies — never claimed as
calibrated production authenticity.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_lock = threading.Lock()
_CACHE: Dict[str, Dict[str, Any]] = {}


def softmax2(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def train_binary_logreg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.25,
    epochs: int = 350,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-class logistic regression. y in {0,1}; returns weights (2,d), bias (2,)."""
    n, d = x.shape
    w = np.zeros((2, d), dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    for _ in range(epochs):
        logits = x @ w.T + b
        probs = softmax2(logits)
        yoh = np.zeros_like(probs)
        yoh[np.arange(n), y.astype(int)] = 1.0
        grad_w = ((probs - yoh).T @ x) / n
        grad_b = (probs - yoh).mean(axis=0)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_proba(vec: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(1, -1)
    # L2 normalize for stability across backbones
    nrm = np.linalg.norm(v) + 1e-12
    v = v / nrm
    logits = v @ weights.T + bias
    return softmax2(logits)[0]


def save_head(
    path: Path,
    meta_path: Path,
    weights: np.ndarray,
    bias: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weights=weights, bias=bias)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with _lock:
        _CACHE.pop(str(path), None)


def load_head(path: Path, meta_path: Path) -> Optional[Dict[str, Any]]:
    key = str(path)
    with _lock:
        if key in _CACHE:
            return _CACHE[key]
        if not path.exists():
            return None
        data = np.load(path)
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {"model_version": "unknown"}
        )
        model = {
            "weights": data["weights"],
            "bias": data["bias"],
            "meta": meta,
        }
        _CACHE[key] = model
        return model


def clear_head_cache() -> None:
    with _lock:
        _CACHE.clear()
