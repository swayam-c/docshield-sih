"""Sensitive-value masking for UI, logs, and stored evidence."""

from __future__ import annotations

import re
from typing import Optional


SENSITIVE_FIELDS = {
    "document_number",
    "aadhaar_number",
    "pan_number",
    "passport_number",
    "visa_number",
    "address",
    "name",
    "father_name",
    "mrz",
    "mrz_line1",
    "mrz_line2",
    "mrz_line3",
}


def mask_value(field: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return text

    key = field.lower()
    if key in {
        "document_number",
        "pan_number",
        "passport_number",
        "aadhaar_number",
        "visa_number",
    }:
        if len(text) <= 4:
            return "X" * len(text)
        return ("X" * (len(text) - 4)) + text[-4:]

    if key == "address":
        if len(text) <= 8:
            return "XXXX"
        return text[:3] + "…" + ("X" * min(8, len(text) - 3))

    if key in {"name", "father_name"}:
        parts = text.split()
        masked = []
        for p in parts:
            if len(p) <= 1:
                masked.append("X")
            else:
                masked.append(p[0] + ("X" * (len(p) - 1)))
        return " ".join(masked)

    if key.startswith("mrz"):
        return re.sub(r"[A-Z0-9]", "X", text.upper())

    if key in {"date_of_birth", "issue_date", "expiry_date"}:
        # Keep structure, mask digits partially
        return re.sub(r"\d", "X", text)

    return text


def is_sensitive(field: str) -> bool:
    return field.lower() in SENSITIVE_FIELDS or field.lower().startswith("mrz")
