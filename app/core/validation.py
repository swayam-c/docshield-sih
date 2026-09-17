"""Secure upload validation — magic bytes, MIME, size, orientation, optional PDF."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings

ALLOWED_MIME: Set[str] = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "application/pdf",
}

EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}

# Executable / script polyglots — never treat as documents
_DANGEROUS_PREFIXES = (
    b"MZ",  # PE/DOS
    b"\x7fELF",
    b"<!DOCTYPE",
    b"<!doctype",
    b"<html",
    b"<HTML",
    b"<?php",
    b"#!/",
)


@dataclass
class ValidatedImage:
    filename: str
    extension: str
    mime_type: str
    size_bytes: int
    width: int
    height: int
    content: bytes  # decoded image bytes (PNG/JPEG/WEBP), never executed
    source_format: str = "IMAGE"  # IMAGE | PDF
    orientation_corrected: bool = False
    pages_rendered: int = 1


class UploadValidationError(Exception):
    def __init__(self, message: str, code: str = "INVALID_UPLOAD"):
        self.message = message
        self.code = code
        super().__init__(message)


def _safe_filename(name: str) -> str:
    base = Path(name or "upload.bin").name
    base = re.sub(r"[^\w.\- ()\[\]]+", "_", base)
    return base.replace("\x00", "")[:200] or "upload.bin"


def _sniff_format(content: bytes) -> Optional[str]:
    if len(content) < 12:
        return None
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if content.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if content[0:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "WEBP"
    if content.startswith(b"%PDF"):
        return "PDF"
    return None


def _reject_dangerous(content: bytes) -> None:
    head = content[:64].lstrip()
    for prefix in _DANGEROUS_PREFIXES:
        if head.startswith(prefix):
            raise UploadValidationError(
                "File appears to be an executable or script, not a document image.",
                "DANGEROUS_CONTENT",
            )


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _load_and_orient_image(content: bytes) -> Tuple[Image.Image, bool]:
    img = Image.open(io.BytesIO(content))
    corrected = False
    try:
        transposed = ImageOps.exif_transpose(img)
        if transposed is not None:
            if transposed is not img:
                corrected = True
            img = transposed
    except Exception:
        pass
    return img.convert("RGB"), corrected


def _render_pdf_first_page(content: bytes, settings: Settings) -> Tuple[bytes, int, int, int]:
    """Render first PDF page to PNG bytes. Requires pypdfium2 when PDF enabled."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise UploadValidationError(
            "PDF support requires pypdfium2. Install dependencies or upload an image.",
            "PDF_UNSUPPORTED",
        ) from exc

    try:
        doc = pdfium.PdfDocument(content)
    except Exception as exc:  # noqa: BLE001
        raise UploadValidationError("Corrupted or unreadable PDF.", "CORRUPT_PDF") from exc

    if len(doc) < 1:
        raise UploadValidationError("PDF has no pages.", "EMPTY_PDF")

    page_count = len(doc)
    page = doc[0]
    # Scale for OCR-friendly resolution without exploding pixels
    scale = 2.0
    bitmap = page.render(scale=scale)
    pil = bitmap.to_pil().convert("RGB")
    w, h = pil.size
    if w * h > settings.max_image_pixels:
        raise UploadValidationError(
            "Rendered PDF page exceeds safe pixel limit.",
            "TOO_MANY_PIXELS",
        )
    return _pil_to_png_bytes(pil), w, h, page_count


def validate_image_bytes(content: bytes, filename: str, settings: Settings) -> ValidatedImage:
    if not content:
        raise UploadValidationError("Empty file.", "EMPTY_FILE")

    if len(content) > settings.max_upload_bytes:
        raise UploadValidationError(
            f"File exceeds size limit of {settings.max_upload_bytes} bytes.",
            "FILE_TOO_LARGE",
        )

    _reject_dangerous(content)

    safe_name = _safe_filename(filename)
    ext = Path(safe_name).suffix.lower()
    allowed = settings.allowed_extension_list
    if ext not in allowed:
        raise UploadValidationError(
            f"Extension '{ext}' not allowed. Allowed: {', '.join(allowed)}",
            "BAD_EXTENSION",
        )

    sniffed = _sniff_format(content)
    if sniffed is None:
        raise UploadValidationError(
            "Unrecognized file signature. Only PNG, JPEG, WEBP"
            + (", PDF" if settings.allow_pdf else "")
            + " are accepted.",
            "BAD_SIGNATURE",
        )

    # Extension vs magic-byte consistency
    if sniffed == "JPEG" and ext not in {".jpg", ".jpeg"}:
        raise UploadValidationError("Extension does not match file signature.", "MIME_MISMATCH")
    if sniffed == "PNG" and ext != ".png":
        raise UploadValidationError("Extension does not match file signature.", "MIME_MISMATCH")
    if sniffed == "WEBP" and ext != ".webp":
        raise UploadValidationError("Extension does not match file signature.", "MIME_MISMATCH")
    if sniffed == "PDF" and ext != ".pdf":
        raise UploadValidationError("Extension does not match file signature.", "MIME_MISMATCH")

    if sniffed == "PDF":
        if not settings.allow_pdf:
            raise UploadValidationError("PDF uploads are disabled.", "PDF_DISABLED")
        png_bytes, width, height, pages = _render_pdf_first_page(content, settings)
        return ValidatedImage(
            filename=safe_name,
            extension=".png",  # stored/processed as rendered PNG
            mime_type="image/png",
            size_bytes=len(png_bytes),
            width=width,
            height=height,
            content=png_bytes,
            source_format="PDF",
            orientation_corrected=False,
            pages_rendered=min(1, pages),
        )

    # Image path: verify decode + orientation
    try:
        with Image.open(io.BytesIO(content)) as probe:
            probe.verify()
        img, oriented = _load_and_orient_image(content)
        width, height = img.size
        pixels = width * height
        if pixels <= 0:
            raise UploadValidationError("Invalid image dimensions.", "BAD_DIMENSIONS")
        if pixels > settings.max_image_pixels:
            raise UploadValidationError(
                "Image resolution exceeds safe pixel limit.",
                "TOO_MANY_PIXELS",
            )
        # Re-encode oriented RGB to PNG for stable downstream pipeline when EXIF fixed
        if oriented:
            out_bytes = _pil_to_png_bytes(img)
            out_ext = ".png"
            out_mime = "image/png"
        else:
            out_bytes = content
            out_ext = ext
            out_mime = EXT_TO_MIME.get(ext, "application/octet-stream")
    except UploadValidationError:
        raise
    except UnidentifiedImageError as exc:
        raise UploadValidationError("Corrupted or unreadable image.", "CORRUPT_IMAGE") from exc
    except OSError as exc:
        raise UploadValidationError("Corrupted or unreadable image.", "CORRUPT_IMAGE") from exc

    if out_mime not in ALLOWED_MIME:
        raise UploadValidationError("MIME type not allowed.", "BAD_MIME")

    return ValidatedImage(
        filename=safe_name,
        extension=out_ext,
        mime_type=out_mime,
        size_bytes=len(out_bytes),
        width=width,
        height=height,
        content=out_bytes,
        source_format="IMAGE",
        orientation_corrected=oriented,
        pages_rendered=1,
    )


async def validate_upload_file(upload: UploadFile, settings: Settings) -> ValidatedImage:
    """Read upload bytes only — never execute uploaded files."""
    content = await upload.read()
    # Declared content-type is advisory; magic bytes are authoritative
    return validate_image_bytes(content, upload.filename or "upload.bin", settings)
