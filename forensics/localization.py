"""SIH Phase 10 — tampering localization / heatmaps (review-only).

Builds coarse spatial anomaly maps from:
  - ELA residual energy
  - High-frequency residual (blur difference)
  - Local noise (Laplacian magnitude)

Does NOT prove fraud. Does NOT set authenticity_estimate.
"""

from __future__ import annotations

import base64
import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def _finite(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _clip01(x: float) -> float:
    return float(np.clip(_finite(x), 0.0, 1.0))


def _normalize_map(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    lo, hi = float(np.percentile(m, 5)), float(np.percentile(m, 95))
    if hi - lo < 1e-6:
        return np.zeros_like(m, dtype=np.float32)
    out = (m - lo) / (hi - lo + 1e-6)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _ela_residual_gray(bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return np.zeros(bgr.shape[:2], dtype=np.float32)
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if recompressed is None:
        return np.zeros(bgr.shape[:2], dtype=np.float32)
    if recompressed.shape != bgr.shape:
        recompressed = cv2.resize(recompressed, (bgr.shape[1], bgr.shape[0]))
    diff = cv2.absdiff(bgr, recompressed).astype(np.float32)
    return cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)


def _hf_residual(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.absdiff(gray, blur).astype(np.float32)


def _noise_mag(gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return np.abs(lap).astype(np.float32)


def _downsample_grid(heat: np.ndarray, rows: int = 24, cols: int = 24) -> List[List[float]]:
    h, w = heat.shape[:2]
    grid: List[List[float]] = []
    for r in range(rows):
        row_vals: List[float] = []
        y0 = int(r * h / rows)
        y1 = int((r + 1) * h / rows)
        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)
            tile = heat[y0:y1, x0:x1]
            val = float(np.mean(tile)) if tile.size else 0.0
            row_vals.append(round(_clip01(val), 4))
        grid.append(row_vals)
    return grid


def _hotspots_from_grid(
    grid: List[List[float]],
    img_w: int,
    img_h: int,
    threshold: float = 0.65,
    max_spots: int = 8,
) -> List[Dict[str, Any]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if not rows or not cols:
        return []
    cells: List[Tuple[float, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            score = float(grid[r][c])
            if score >= threshold:
                cells.append((score, r, c))
    cells.sort(reverse=True)
    spots: List[Dict[str, Any]] = []
    for score, r, c in cells[:max_spots]:
        x = int(c * img_w / cols)
        y = int(r * img_h / rows)
        bw = max(1, int(img_w / cols))
        bh = max(1, int(img_h / rows))
        spots.append(
            {
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "score": round(score, 4),
                "grid_row": r,
                "grid_col": c,
            }
        )
    return spots


def _overlay_png_b64(bgr: np.ndarray, heat: np.ndarray, max_side: int = 480) -> str:
    """Colorize heat and blend over a downscaled preview. Returns base64 PNG (no data: prefix)."""
    h, w = bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        preview = cv2.resize(bgr, (nw, nh))
        heat_s = cv2.resize(heat, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        preview = bgr
        heat_s = heat

    heat_u8 = (np.clip(heat_s, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(preview, 0.55, colored, 0.45, 0)
    ok, buf = cv2.imencode(".png", blended)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_localization(bgr: np.ndarray) -> Dict[str, Any]:
    """Compute combined anomaly heatmap + coarse grid + hotspots."""
    if bgr is None or bgr.size == 0:
        return {
            "status": "ERROR",
            "error": "Empty image for localization.",
            "note": "Localization not assessed.",
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    ela = _normalize_map(_ela_residual_gray(bgr))
    hf = _normalize_map(_hf_residual(gray))
    noise = _normalize_map(_noise_mag(gray))

    # Combined review heatmap (not authenticity probability)
    combined = _normalize_map(0.45 * ela + 0.30 * hf + 0.25 * noise)
    # Mild blur so UI looks like a region map, not salt noise
    combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=max(1.0, min(h, w) / 80.0))
    combined = _normalize_map(combined)

    grid = _downsample_grid(combined, rows=24, cols=24)
    hotspots = _hotspots_from_grid(grid, w, h, threshold=0.65, max_spots=8)
    overlay_b64 = _overlay_png_b64(bgr, combined)

    peak = float(np.max(combined)) if combined.size else 0.0
    mean = float(np.mean(combined)) if combined.size else 0.0
    coverage = float(np.mean(combined >= 0.65)) if combined.size else 0.0

    return {
        "status": "LOCALIZED",
        "method": "ela_hf_noise_blend_v1",
        "grid_rows": 24,
        "grid_cols": 24,
        "grid": grid,
        "hotspots": hotspots,
        "hotspot_count": len(hotspots),
        "peak_score": round(_clip01(peak), 4),
        "mean_score": round(_clip01(mean), 4),
        "high_anomaly_coverage": round(_clip01(coverage), 4),
        "overlay_png_base64": overlay_b64 or None,
        "overlay_mime": "image/png" if overlay_b64 else None,
        "components": {
            "ela_weight": 0.45,
            "hf_residual_weight": 0.30,
            "noise_weight": 0.25,
        },
        "concern": "HEURISTIC_LOCALIZATION_ONLY",
        "calibrated": False,
        "note": (
            "Heatmap / hotspots are spatial review cues derived from classical residuals. "
            "They are NOT pixel-proof of tampering and do not set authenticity_estimate."
        ),
    }
