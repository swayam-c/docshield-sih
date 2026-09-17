"""Document-type classification — trainable logistic head + heuristic fallback.

SIH core classes: PASSPORT, VISA, NATIONAL_ID, DRIVING_LICENCE, PERMIT, OTHER, UNKNOWN.
Indian subtypes PAN/AADHAAR may refine NATIONAL_ID when cues are strong.

The logistic model is trained on synthetic feature distributions (authorized synthetic
proxy) — not on real government document dumps. Report method honestly.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from document_profiles import (
    DOCUMENT_TYPES,
    PROFILE_BY_TYPE,
    SIH_CORE_TYPES,
    to_sih_category,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "document_classifier"
MODEL_PATH = MODEL_DIR / "doc_type_logreg.npz"
META_PATH = MODEL_DIR / "doc_type_logreg_meta.json"

FEATURE_NAMES = (
    "aspect",
    "card_like",
    "tall_page",
    "blue_frac",
    "warm_frac",
    "cyan_frac",
    "mean_sat_n",
    "mean_val_n",
    "top_edge",
    "mid_edge",
    "bottom_edge",
    "photo_left_bias",
    "prior_passport",
    "prior_visa",
    "prior_national_id",
    "prior_dl",
    "prior_permit",
    "prior_pan",
    "prior_aadhaar",
)

_lock = threading.Lock()
_model_cache: Dict[str, Any] = {}


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    sih_category: str
    alternatives: List[Dict[str, Any]]
    method: str
    features: Dict[str, float]
    profile: str
    model_version: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # API / UI aliases (keep legacy keys too)
        data["document_type"] = (self.label or "").lower()
        data["document_type_confidence"] = self.confidence
        data["document_type_label"] = self.label
        return data


def _decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image for classification.")
    return img


def _aspect_ratio(w: int, h: int) -> float:
    if min(w, h) <= 0:
        return 1.0
    return max(w, h) / min(w, h)


def _filename_priors(filename: str) -> Dict[str, float]:
    name = (filename or "").lower()
    priors = {t: 0.0 for t in DOCUMENT_TYPES}
    patterns = {
        "PAN": [r"\bpan\b", r"permanent.?account"],
        "AADHAAR": [r"aadhaar", r"aadhar", r"uidai", r"\buid\b"],
        "PASSPORT": [r"passport", r"\bppt\b"],
        "VISA": [r"\bvisa\b"],
        "DRIVING_LICENCE": [r"driving", r"\bdl\b", r"licence", r"license"],
        "PERMIT": [r"permit", r"work.?permit", r"residence.?permit"],
        "NATIONAL_ID": [r"national.?id", r"\bnid\b", r"identity.?card", r"\bid.?card\b"],
    }
    for label, pats in patterns.items():
        if any(re.search(p, name) for p in pats):
            priors[label] = 0.35
    if not any(priors.values()):
        priors["OTHER"] = 0.05
    return priors


def _color_cues(bgr: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return {
        "mean_saturation": float(np.mean(s)),
        "mean_value": float(np.mean(v)),
        "blue_frac": float(np.mean((h > 90) & (h < 130) & (s > 40))),
        "warm_frac": float(np.mean(((h < 35) | (h > 150)) & (s > 20) & (v > 120))),
        "cyan_frac": float(np.mean((h > 75) & (h < 100) & (s > 30))),
        "green_frac": float(np.mean((h > 35) & (h < 85) & (s > 40))),
    }


def _layout_cues(gray: np.ndarray) -> Dict[str, float]:
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    row_energy = edges.mean(axis=1)
    top = float(np.mean(row_energy[: max(1, h // 5)]))
    mid = float(np.mean(row_energy[h // 3 : 2 * h // 3]))
    bottom = float(np.mean(row_energy[4 * h // 5 :]))
    left = gray[:, : w // 3]
    right = gray[:, 2 * w // 3 :]
    left_var = float(np.var(left))
    right_var = float(np.var(right))
    return {
        "top_edge_energy": top,
        "mid_edge_energy": mid,
        "bottom_edge_energy": bottom,
        "left_var": left_var,
        "right_var": right_var,
        "photo_left_bias": float(
            np.clip((right_var - left_var) / (right_var + left_var + 1e-6), -1, 1)
        ),
    }


def extract_feature_dict(
    image_bytes: bytes, filename: str = ""
) -> Tuple[Dict[str, float], Dict[str, float]]:
    bgr = _decode_bgr(image_bytes)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    aspect = _aspect_ratio(w, h)
    colors = _color_cues(bgr)
    layout = _layout_cues(gray)
    priors = _filename_priors(filename)
    card_like = float(np.exp(-((aspect - 1.586) ** 2) / (2 * 0.12**2)))
    tall_page = float(np.exp(-((aspect - 1.42) ** 2) / (2 * 0.18**2)))

    feats = {
        "aspect": aspect,
        "card_like": card_like,
        "tall_page": tall_page,
        "blue_frac": colors["blue_frac"],
        "warm_frac": colors["warm_frac"],
        "cyan_frac": colors["cyan_frac"],
        "mean_sat_n": colors["mean_saturation"] / 255.0,
        "mean_val_n": colors["mean_value"] / 255.0,
        "top_edge": layout["top_edge_energy"],
        "mid_edge": layout["mid_edge_energy"],
        "bottom_edge": layout["bottom_edge_energy"],
        "photo_left_bias": layout["photo_left_bias"],
        "prior_passport": priors.get("PASSPORT", 0.0),
        "prior_visa": priors.get("VISA", 0.0),
        "prior_national_id": max(priors.get("NATIONAL_ID", 0.0), priors.get("AADHAAR", 0.0) * 0.5),
        "prior_dl": priors.get("DRIVING_LICENCE", 0.0),
        "prior_permit": priors.get("PERMIT", 0.0),
        "prior_pan": priors.get("PAN", 0.0),
        "prior_aadhaar": priors.get("AADHAAR", 0.0),
        "width": float(w),
        "height": float(h),
        "green_frac": colors["green_frac"],
    }
    return feats, priors


def feature_vector(feats: Dict[str, float]) -> np.ndarray:
    return np.array([float(feats.get(n, 0.0)) for n in FEATURE_NAMES], dtype=np.float64)


def _heuristic_core_scores(feats: Dict[str, float], priors: Dict[str, float]) -> Dict[str, float]:
    scores = {t: 0.05 for t in SIH_CORE_TYPES}
    scores["PASSPORT"] += 0.28 * max(feats["tall_page"], 1.0 - abs(feats["aspect"] - 1.35) / 2)
    scores["PASSPORT"] += 0.25 * feats["blue_frac"]
    scores["PASSPORT"] += 0.12 * max(0.0, feats["photo_left_bias"])
    if feats["bottom_edge"] > feats["mid_edge"] * 1.05:
        scores["PASSPORT"] += 0.12

    scores["VISA"] += 0.22 * feats["card_like"]
    scores["VISA"] += 0.18 * feats["green_frac"] if "green_frac" in feats else 0.0
    scores["VISA"] += 0.20 * feats["prior_visa"]
    scores["VISA"] += 0.10 * feats["warm_frac"]

    scores["NATIONAL_ID"] += 0.32 * feats["card_like"]
    scores["NATIONAL_ID"] += 0.18 * feats["cyan_frac"] + 0.10 * feats["warm_frac"]
    scores["NATIONAL_ID"] += 0.12 * max(0.0, feats["photo_left_bias"])
    scores["NATIONAL_ID"] += 0.15 * max(feats["prior_national_id"], feats["prior_aadhaar"], feats["prior_pan"])

    scores["DRIVING_LICENCE"] += 0.30 * feats["card_like"]
    scores["DRIVING_LICENCE"] += 0.12 * feats["blue_frac"]
    scores["DRIVING_LICENCE"] += 0.12 * max(0.0, feats["photo_left_bias"])
    scores["DRIVING_LICENCE"] += 0.25 * feats["prior_dl"]

    scores["PERMIT"] += 0.20 * feats["card_like"]
    scores["PERMIT"] += 0.15 * feats["warm_frac"]
    scores["PERMIT"] += 0.30 * feats["prior_permit"]

    scores["OTHER"] += 0.12
    scores["UNKNOWN"] += 0.08

    for k, v in (
        ("PASSPORT", feats["prior_passport"]),
        ("VISA", feats["prior_visa"]),
        ("NATIONAL_ID", feats["prior_national_id"]),
        ("DRIVING_LICENCE", feats["prior_dl"]),
        ("PERMIT", feats["prior_permit"]),
    ):
        scores[k] += v

    arr = np.array([scores[t] for t in SIH_CORE_TYPES], dtype=np.float64)
    arr = np.clip(arr, 1e-6, None)
    arr = arr / arr.sum()
    return {t: float(arr[i]) for i, t in enumerate(SIH_CORE_TYPES)}


def _refine_subtype(core: str, feats: Dict[str, float], priors: Dict[str, float]) -> str:
    if core != "NATIONAL_ID":
        return core
    pan_s = feats["prior_pan"] + 0.15 * feats["warm_frac"] + 0.1 * max(0.0, -feats["photo_left_bias"])
    aad_s = feats["prior_aadhaar"] + 0.2 * feats["cyan_frac"] + 0.1 * max(0.0, feats["photo_left_bias"])
    if pan_s > aad_s and pan_s > 0.25:
        return "PAN"
    if aad_s > pan_s and aad_s > 0.25:
        return "AADHAAR"
    return "NATIONAL_ID"


# --- Synthetic-trained multinomial logistic regression (numpy) ---


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _train_logreg(
    x: np.ndarray, y: np.ndarray, n_classes: int, lr: float = 0.35, epochs: int = 400
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    w = np.zeros((n_classes, d), dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)
    for _ in range(epochs):
        logits = x @ w.T + b
        probs = _softmax(logits)
        yoh = np.zeros_like(probs)
        yoh[np.arange(n), y] = 1.0
        grad_w = ((probs - yoh).T @ x) / n
        grad_b = (probs - yoh).mean(axis=0)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def _class_centroids() -> Dict[str, np.ndarray]:
    """Synthetic class means in FEATURE_NAMES space (authorized synthetic proxy)."""
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    base = np.zeros(len(FEATURE_NAMES))
    cents = {}

    def setc(name: str, **kwargs: float) -> None:
        v = base.copy()
        for k, val in kwargs.items():
            v[idx[k]] = val
        cents[name] = v

    setc(
        "PASSPORT",
        aspect=1.40,
        card_like=0.35,
        tall_page=0.85,
        blue_frac=0.45,
        photo_left_bias=0.4,
        bottom_edge=0.12,
        prior_passport=0.2,
    )
    setc(
        "VISA",
        aspect=1.50,
        card_like=0.7,
        tall_page=0.3,
        warm_frac=0.25,
        prior_visa=0.25,
        mean_val_n=0.65,
    )
    setc(
        "NATIONAL_ID",
        aspect=1.586,
        card_like=0.9,
        cyan_frac=0.3,
        warm_frac=0.2,
        photo_left_bias=0.35,
        prior_national_id=0.2,
    )
    setc(
        "DRIVING_LICENCE",
        aspect=1.586,
        card_like=0.88,
        blue_frac=0.2,
        photo_left_bias=0.35,
        prior_dl=0.25,
    )
    setc(
        "PERMIT",
        aspect=1.45,
        card_like=0.55,
        warm_frac=0.35,
        prior_permit=0.3,
        mean_sat_n=0.35,
    )
    setc("OTHER", aspect=1.2, card_like=0.3, tall_page=0.2, mean_val_n=0.5)
    setc("UNKNOWN", aspect=1.0, card_like=0.15, tall_page=0.15, mean_val_n=0.4)
    return cents


def generate_synthetic_training_set(
    samples_per_class: int = 80, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    cents = _class_centroids()
    labels = list(SIH_CORE_TYPES)
    xs, ys = [], []
    for yi, lab in enumerate(labels):
        mean = cents[lab]
        for _ in range(samples_per_class):
            noise = rng.normal(0.0, 0.08, size=mean.shape)
            # filename prior spikes occasionally
            sample = np.clip(mean + noise, -1.0, 1.5)
            xs.append(sample)
            ys.append(yi)
    return np.asarray(xs), np.asarray(ys), labels


def train_and_save_classifier(samples_per_class: int = 80) -> Dict[str, Any]:
    x, y, labels = generate_synthetic_training_set(samples_per_class=samples_per_class)
    w, b = _train_logreg(x, y, n_classes=len(labels))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_PATH, weights=w, bias=b, labels=np.array(labels))
    meta = {
        "model_name": "document_type_logreg",
        "model_version": "sih-phase2-synthetic-v1",
        "dataset": "synthetic_feature_centroids_v1",
        "classes": labels,
        "feature_names": list(FEATURE_NAMES),
        "samples_per_class": samples_per_class,
        "note": (
            "Trained on synthetic feature distributions only — "
            "not on real government identity documents."
        ),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with _lock:
        _model_cache.clear()
    return meta


def _load_model() -> Optional[Dict[str, Any]]:
    with _lock:
        if "model" in _model_cache:
            return _model_cache["model"]
        if not MODEL_PATH.exists():
            return None
        data = np.load(MODEL_PATH, allow_pickle=True)
        model = {
            "weights": data["weights"],
            "bias": data["bias"],
            "labels": [str(x) for x in data["labels"].tolist()],
            "meta": json.loads(META_PATH.read_text(encoding="utf-8"))
            if META_PATH.exists()
            else {"model_version": "unknown"},
        }
        _model_cache["model"] = model
        return model


def ensure_trained_model() -> Dict[str, Any]:
    model = _load_model()
    if model is not None:
        return model["meta"] if "meta" in model else {"model_version": "loaded"}
    return train_and_save_classifier()


def _predict_logreg(vec: np.ndarray, model: Dict[str, Any]) -> Dict[str, float]:
    logits = vec @ model["weights"].T + model["bias"]
    probs = _softmax(logits.reshape(1, -1))[0]
    return {lab: float(probs[i]) for i, lab in enumerate(model["labels"])}


def classify_document_type(image_bytes: bytes, filename: str = "") -> ClassificationResult:
    feats, priors = extract_feature_dict(image_bytes, filename=filename)
    vec = feature_vector(feats)

    heuristic = _heuristic_core_scores(feats, priors)
    method = "heuristic_v2_geometry_color_layout"
    model_version = "none"
    probs = heuristic

    model = _load_model()
    if model is None:
        # Auto-fit synthetic model once for SIH demo reproducibility
        try:
            train_and_save_classifier()
            model = _load_model()
        except Exception:
            model = None

    if model is not None:
        ml_probs = _predict_logreg(vec, model)
        # Blend trained synthetic head with transparent heuristic
        probs = {
            t: 0.55 * ml_probs.get(t, 0.0) + 0.45 * heuristic.get(t, 0.0)
            for t in SIH_CORE_TYPES
        }
        s = sum(probs.values()) or 1.0
        probs = {k: v / s for k, v in probs.items()}
        method = "logreg_synthetic_v1+heuristic_v2"
        model_version = model.get("meta", {}).get("model_version", "sih-phase2-synthetic-v1")

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    core_label, confidence = ranked[0]
    alternatives = [
        {"label": lab, "confidence": round(conf, 4)} for lab, conf in ranked[1:4] if conf > 0.04
    ]

    # Low margin / low confidence → UNKNOWN (OOD-friendly)
    if confidence < 0.28 or (len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.05 and confidence < 0.4):
        core_label = "UNKNOWN"
        confidence = min(confidence, 0.4)
        method = method + "+unknown_gate"

    label = _refine_subtype(core_label, feats, priors)
    if label not in PROFILE_BY_TYPE:
        label = core_label

    profile = PROFILE_BY_TYPE[label].profile_id
    feature_out = {k: round(float(v), 4) for k, v in feats.items() if k in FEATURE_NAMES or k in {"width", "height", "green_frac"}}

    return ClassificationResult(
        label=label,
        confidence=round(float(confidence), 4),
        sih_category=to_sih_category(label),
        alternatives=alternatives,
        method=method,
        features=feature_out,
        profile=profile,
        model_version=model_version,
    )
