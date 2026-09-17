"""Lazy-loaded model registry — load each backbone once per process."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

from app.ml.device import get_device, use_mock_backends

_lock = threading.Lock()
_CACHE: Dict[str, Any] = {}


def clear_registry() -> None:
    with _lock:
        _CACHE.clear()


def _load_efficientnet():
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(get_device())
    return model, weights.transforms()


def _load_vit():
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.heads = torch.nn.Identity()
    model.eval()
    model.to(get_device())
    return model, weights.transforms()


def _load_clip():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    model.to(get_device())
    return model, preprocess


def get_backbone(name: str) -> Tuple[Any, Any]:
    """Return (model, preprocess). Raises if unavailable."""
    if use_mock_backends():
        raise RuntimeError("Mock mode — no real backbone load.")

    with _lock:
        if name in _CACHE:
            return _CACHE[name]
        if name == "efficientnet_b0":
            pair = _load_efficientnet()
        elif name == "vit_b16":
            pair = _load_vit()
        elif name == "clip_vit_b32":
            pair = _load_clip()
        else:
            raise KeyError(f"Unknown backbone: {name}")
        _CACHE[name] = pair
        return pair


def clip_available() -> bool:
    if use_mock_backends():
        return False
    try:
        import open_clip  # noqa: F401

        return True
    except ImportError:
        return False


@torch.inference_mode()
def forward_image(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    device = get_device()
    tensor = tensor.to(device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    out = model(tensor)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.squeeze(0).detach().float().cpu()


def pil_from_bytes(image_bytes: bytes) -> Image.Image:
    from io import BytesIO

    return Image.open(BytesIO(image_bytes)).convert("RGB")
