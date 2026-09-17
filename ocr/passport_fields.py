"""Passport-oriented OCR field helpers (Surname / Given Names layouts).

Never fabricates values. OCR confidence is transcription-only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ocr.field_patterns import (
    DATE_RE,
    DOB_LABEL_RE,
    EXPIRY_RE,
    ISSUE_RE,
    NATIONALITY_RE,
    PASSPORT_LABEL_RE,
    PASSPORT_RE,
    SEX_RE,
)

SURNAME_RE = re.compile(
    r"(?:surname(?:\s*/\s*nom)?|nom)\s*[:/\-]?\s*(?:\n|\r\n)?\s*"
    r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})",
    re.I,
)
GIVEN_NAMES_RE = re.compile(
    r"(?:given\s*names?(?:\s*/\s*prenoms?)?|prenoms?|forenames?)\s*[:/\-]?\s*(?:\n|\r\n)?\s*"
    r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})",
    re.I,
)
NAME_COLON_RE = re.compile(
    r"(?<!father\s)(?<!mother\s)(?:full\s*)?name\s*[:\-]\s*"
    r"([A-Z][A-Za-z]+(?:[\s,]+[A-Z][A-Za-z]+){0,4})",
    re.I,
)


def _first(patterns: List[re.Pattern], text: str) -> Optional[str]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return None


def extract_passport_name(text: str) -> Optional[str]:
    surname = _first([SURNAME_RE], text)
    given = _first([GIVEN_NAMES_RE], text)
    if surname and given:
        return f"{surname} {given}".strip()
    if given:
        return given
    if surname:
        return surname
    return _first([NAME_COLON_RE], text)


def extract_passport_number(text: str) -> Optional[str]:
    return _first([PASSPORT_LABEL_RE, PASSPORT_RE], text.upper())


def extract_passport_fields(
    full_text: str,
    text_boxes: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Build passport field candidates using passport-specific labels.
    Returns (fields_list_raw_candidates, raw_by_name) without status reconciliation —
    callers should route through ocr.fields._field for confidence/status.
    """
    # Deferred import avoids circular dependency at module load
    from ocr.fields import _field, _validate_document_number
    from ocr.dates import validate_date_field

    text = full_text or ""
    fields: List[Dict[str, Any]] = []
    raw: Dict[str, str] = {}

    name = extract_passport_name(text)
    if name:
        fields.append(_field("name", re.sub(r"\s+", " ", name), text_boxes, "PASS"))
        raw["name"] = name

    doc = extract_passport_number(text)
    if doc:
        fields.append(
            _field(
                "document_number",
                doc,
                text_boxes,
                _validate_document_number("PASSPORT", doc),
            )
        )
        raw["document_number"] = doc
        # Alias for UI/tests that still look for passport_number
        fields.append(
            _field(
                "passport_number",
                doc,
                text_boxes,
                _validate_document_number("PASSPORT", doc),
            )
        )
        raw["passport_number"] = doc

    nat = _first([NATIONALITY_RE], text)
    if nat:
        fields.append(_field("nationality", nat.upper()[:20], text_boxes, "PASS"))
        raw["nationality"] = nat

    sex_m = SEX_RE.search(text)
    if sex_m:
        sex = sex_m.group(1).upper()
        if sex.startswith("M"):
            sex = "M"
        elif sex.startswith("F"):
            sex = "F"
        fields.append(_field("sex", sex, text_boxes, "PASS"))
        raw["sex"] = sex

    dob = _first([DOB_LABEL_RE], text) or (
        DATE_RE.search(text).group(1) if DATE_RE.search(text) else None
    )
    if dob:
        info = validate_date_field("date_of_birth", dob)
        fields.append(
            _field(
                "date_of_birth",
                info.get("parsed_iso") or dob,
                text_boxes,
                info["status"],
                raw_value=dob,
            )
        )
        raw["date_of_birth"] = dob

    issue = _first([ISSUE_RE], text)
    if issue:
        info = validate_date_field("issue_date", issue)
        fields.append(
            _field(
                "issue_date",
                info.get("parsed_iso") or issue,
                text_boxes,
                info["status"],
                raw_value=issue,
            )
        )
        raw["issue_date"] = issue

    expiry = _first([EXPIRY_RE], text)
    if expiry:
        info = validate_date_field("expiry_date", expiry)
        fields.append(
            _field(
                "expiry_date",
                info.get("parsed_iso") or expiry,
                text_boxes,
                info["status"],
                raw_value=expiry,
            )
        )
        raw["expiry_date"] = expiry

    return fields, raw
