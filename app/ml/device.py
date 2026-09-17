"""Shared device + ML mode helpers."""

from __future__ import annotations

import os
from functools import lru_cache

import torch


def ml_mode() -> str:
    """auto | real | mock — mock skips weight downloads (tests)."""
    return os.getenv("DOCSHIELD_ML_MODE", "auto").strip().lower()


@lru_cache
def get_device() -> torch.device:
    if torch.cuda.is_available() and os.getenv("DOCSHIELD_FORCE_CPU", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return torch.device("cuda")
    return torch.device("cpu")


def use_mock_backends() -> bool:
    mode = ml_mode()
    if mode == "mock":
        return True
    if mode == "real":
        return False
    # auto: mock when env requests or when explicitly testing
    return os.getenv("DOCSHIELD_ML_MOCK", "").lower() in {"1", "true", "yes"}
