"""Field extraction from OCR text using document profiles + patterns.

Patterns use publicly known format shapes only (e.g. PAN ABCDE1234F).
OCR confidence must never be treated as authenticity confidence.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from document_profiles import get_profile
from ocr.dates import parse_date, validate_date_field
from ocr.field_patterns import (
    AADHAAR_RE,
    ADDRESS_RE,
    DATE_RE,
    DL_LABEL_RE,
    DL_RE,
    DOB_LABEL_RE,
    DURATION_RE,
    ENTRIES_RE,
    EXPIRY_RE,
    FATHER_NAME_RE,
    GENERIC_ID_RE,
    ISSUE_RE,
    NAME_LABEL_RE,
    NATIONALITY_RE,
    PAN_LABEL_RE,
    PAN_RE,
    PASSPORT_LABEL_RE,
    PASSPORT_RE,
    PERMIT_TYPE_RE,
    SEX_RE,
    VISA_NUMBER_RE,
    VISA_TYPE_RE,
)
from ocr.masking import mask_value

# Visual-only / handled elsewhere — never counted in OCR completeness
NON_OCR_EXPECTED = {"photo", "signature", "mrz"}

# OCR confidence gates (transcription only)
CONF_EXTRACTED = 0.35
CONF_UNCERTAIN = 0.15


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def _bbox_for_token(token: str, text_boxes: List[Dict[str, Any]]) -> Optional[List[int]]:
    token_u = (token or "").upper().replace(" ", "")
    if not token_u:
        return None
    for box in text_boxes:
        t = str(box.get("text", "")).upper().replace(" ", "")
        if token_u and (token_u in t or t in token_u):
            return box.get("bbox")
    for part in re.findall(r"[A-Za-z0-9]+", token):
        for box in text_boxes:
            if part.upper() == str(box.get("text", "")).upper():
                return box.get("bbox")
    return None


def _source_region(bbox: Optional[List[int]]) -> Optional[Dict[str, int]]:
    if not bbox or len(bbox) < 4:
        return None
    x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    return {"x": x, "y": y, "w": w, "h": h, "x2": x + w, "y2": y + h}


def _confidence_for_token(token: str, text_boxes: List[Dict[str, Any]]) -> float:
    """Return OCR confidence only from boxes that match the token — never invent mean conf."""
    confs = []
    parts = re.findall(r"[A-Za-z0-9]+", token or "")
    if not parts or not text_boxes:
        return 0.0
    for part in parts:
        for box in text_boxes:
            if str(box.get("text", "")).upper() == part.upper():
                confs.append(_safe_float(box.get("confidence", 0)))
    if not confs:
        # Partial containment match
        token_u = (token or "").upper().replace(" ", "")
        for box in text_boxes:
            t = str(box.get("text", "")).upper().replace(" ", "")
            if token_u and t and (token_u in t or t in token_u):
                confs.append(_safe_float(box.get("confidence", 0)))
    if not confs:
        return 0.0
    return round(sum(confs) / len(confs), 4)


def _validate_document_number(doc_type: str, value: str) -> str:
    v = value.replace(" ", "").upper()
    if doc_type == "PAN":
        return "PASS" if PAN_RE.fullmatch(v) else "WARNING"
    if doc_type == "AADHAAR":
        digits = re.sub(r"\D", "", v)
        return "PASS" if len(digits) == 12 else "WARNING"
    if doc_type == "PASSPORT":
        return "PASS" if PASSPORT_RE.fullmatch(v) else "WARNING"
    if doc_type == "DRIVING_LICENCE":
        return "PASS" if len(re.sub(r"[\s\-]", "", v)) >= 8 else "WARNING"
    if doc_type in {"VISA", "PERMIT", "NATIONAL_ID"}:
        return "PASS" if len(v) >= 6 else "WARNING"
    return "PASS" if len(v) >= 4 else "WARNING"


def _derive_status(ocr_confidence: float, validation: str) -> str:
    """Map confidence + format validation → field status (never PASS with 0% OCR)."""
    if ocr_confidence < CONF_UNCERTAIN:
        return "UNCERTAIN"
    if ocr_confidence < CONF_EXTRACTED:
        return "UNCERTAIN"
    if validation == "FAIL":
        return "UNCERTAIN"
    return "EXTRACTED"


def _reconcile_validation(ocr_confidence: float, validation: str) -> str:
    """Format PASS is invalid when OCR confidence is effectively missing/zero."""
    if ocr_confidence < CONF_UNCERTAIN and validation == "PASS":
        return "WARNING"
    if ocr_confidence < CONF_EXTRACTED and validation == "PASS":
        return "WARNING"
    return validation


def _field(
    name: str,
    value: str,
    text_boxes: List[Dict[str, Any]],
    validation: str,
    *,
    extra: Optional[Dict] = None,
    raw_value: Optional[str] = None,
) -> Dict[str, Any]:
    raw = raw_value or value
    bbox = _bbox_for_token(raw, text_boxes)
    ocr_confidence = _confidence_for_token(raw, text_boxes)
    validation = _reconcile_validation(ocr_confidence, validation)
    status = _derive_status(ocr_confidence, validation)
    boxed = {
        "field": name,
        "value": mask_value(name, value),
        "value_present": bool(value),
        "ocr_confidence": ocr_confidence,
        "confidence": ocr_confidence,  # alias per Phase 1–8 contract
        "bounding_box": bbox,
        "bbox": bbox,
        "source_region": _source_region(bbox),
        "validation": validation,
        "status": status,
        "note": "ocr_confidence is transcription confidence, not authenticity.",
    }
    if extra:
        boxed.update(extra)
    return boxed


def _first_match(patterns: List[re.Pattern], text: str) -> Optional[str]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return None


def _extract_document_number(document_type: str, text: str) -> Optional[str]:
    upper = text.upper()
    if document_type == "PAN":
        return _first_match([PAN_LABEL_RE, PAN_RE], upper)
    if document_type == "AADHAAR":
        m = AADHAAR_RE.search(text)
        return m.group(1) if m else None
    if document_type == "PASSPORT":
        return _first_match([PASSPORT_LABEL_RE, PASSPORT_RE], upper)
    if document_type == "DRIVING_LICENCE":
        return _first_match([DL_LABEL_RE, DL_RE], upper)
    if document_type == "VISA":
        return _first_match([VISA_NUMBER_RE, GENERIC_ID_RE, PASSPORT_RE], upper)
    if document_type == "PERMIT":
        return _first_match([GENERIC_ID_RE, PASSPORT_RE, PAN_RE], upper)
    if document_type == "NATIONAL_ID":
        aad = AADHAAR_RE.search(text)
        if aad:
            return aad.group(1)
        return _first_match([PAN_RE, GENERIC_ID_RE], upper)
    aad = AADHAAR_RE.search(text)
    if aad:
        return aad.group(1)
    return _first_match([PAN_RE, PASSPORT_RE, DL_RE], upper)


def _cross_field_checks(fields: List[Dict[str, Any]], raw_by_name: Dict[str, str]) -> List[Dict[str, Any]]:
    """Logical date ordering only — never used as authenticity proof."""
    checks: List[Dict[str, Any]] = []
    dob_raw = raw_by_name.get("date_of_birth")
    issue_raw = raw_by_name.get("issue_date")
    expiry_raw = raw_by_name.get("expiry_date")
    dob, _ = parse_date(dob_raw) if dob_raw else (None, "")
    issue, _ = parse_date(issue_raw) if issue_raw else (None, "")
    expiry, _ = parse_date(expiry_raw) if expiry_raw else (None, "")

    if dob and issue and issue < dob:
        checks.append(
            {
                "check": "issue_after_dob",
                "status": "FAIL",
                "note": "Issue date precedes date of birth (transcription/layout issue possible).",
            }
        )
    if issue and expiry and expiry < issue:
        checks.append(
            {
                "check": "expiry_after_issue",
                "status": "FAIL",
                "note": "Expiry precedes issue date (transcription/layout issue possible).",
            }
        )
    if dob and expiry and expiry < dob:
        checks.append(
            {
                "check": "expiry_after_dob",
                "status": "WARNING",
                "note": "Expiry precedes DOB — likely OCR error.",
            }
        )
    if not checks and (dob or issue or expiry):
        checks.append(
            {
                "check": "date_ordering",
                "status": "PASS",
                "note": "No date-ordering conflicts among extracted dates.",
            }
        )
    return checks


def _applicable_ocr_fields(document_type: str) -> List[str]:
    profile = get_profile(document_type)
    return [f for f in profile.expected_fields if f not in NON_OCR_EXPECTED]


def extract_fields(
    full_text: str,
    text_boxes: List[Dict[str, Any]],
    document_type: str,
) -> Dict[str, Any]:
    profile = get_profile(document_type)
    text = full_text or ""
    fields: List[Dict[str, Any]] = []
    raw_by_name: Dict[str, str] = {}

    # Only extract father_name for profiles that expect it
    expect_father = "father_name" in profile.expected_fields

    # Passport: use dedicated Surname/Given Names extractors first
    if document_type == "PASSPORT":
        from ocr.passport_fields import extract_passport_fields

        p_fields, p_raw = extract_passport_fields(text, text_boxes)
        fields.extend(p_fields)
        raw_by_name.update(p_raw)

    doc_val = None
    if "document_number" not in raw_by_name and "visa_number" not in raw_by_name:
        doc_val = _extract_document_number(document_type, text)
    if doc_val:
        field_name = "document_number"
        if document_type == "VISA":
            visa_m = VISA_NUMBER_RE.search(text)
            if visa_m:
                fields.append(
                    _field(
                        "visa_number",
                        visa_m.group(1).strip(),
                        text_boxes,
                        _validate_document_number("VISA", visa_m.group(1)),
                    )
                )
                raw_by_name["visa_number"] = visa_m.group(1).strip()
                if "document_number" not in profile.expected_fields:
                    doc_val = None
        if doc_val:
            fields.append(
                _field(
                    field_name,
                    doc_val,
                    text_boxes,
                    _validate_document_number(document_type, doc_val),
                )
            )
            raw_by_name[field_name] = doc_val

    if "name" not in raw_by_name:
        m = NAME_LABEL_RE.search(text)
        if m:
            name = re.sub(r"\s+", " ", m.group(1).strip().replace(",", " "))
            fields.append(_field("name", name, text_boxes, "PASS"))
            raw_by_name["name"] = name

    if expect_father:
        m = FATHER_NAME_RE.search(text)
        if m:
            fname = re.sub(r"\s+", " ", m.group(1).strip())
            fields.append(_field("father_name", fname, text_boxes, "PASS"))
            raw_by_name["father_name"] = fname

    if "date_of_birth" not in raw_by_name:
        m = DOB_LABEL_RE.search(text)
        dob_val = m.group(1) if m else None
        dob_unlabeled = False
        if not dob_val and "date_of_birth" in profile.expected_fields:
            dm = DATE_RE.search(text)
            dob_val = dm.group(1) if dm else None
            dob_unlabeled = bool(dob_val)
        if dob_val:
            date_info = validate_date_field("date_of_birth", dob_val)
            validation = date_info["status"] if not dob_unlabeled else (
                "WARNING" if date_info["status"] == "PASS" else date_info["status"]
            )
            fields.append(
                _field(
                    "date_of_birth",
                    dob_val,
                    text_boxes,
                    validation,
                    extra={"date_checks": date_info, "label_matched": not dob_unlabeled},
                )
            )
            raw_by_name["date_of_birth"] = dob_val

    if "issue_date" not in raw_by_name:
        m = ISSUE_RE.search(text)
        if m:
            date_info = validate_date_field("issue_date", m.group(1))
            fields.append(
                _field(
                    "issue_date",
                    m.group(1),
                    text_boxes,
                    date_info["status"],
                    extra={"date_checks": date_info},
                )
            )
            raw_by_name["issue_date"] = m.group(1)

    if "expiry_date" not in raw_by_name:
        m = EXPIRY_RE.search(text)
        if m:
            date_info = validate_date_field("expiry_date", m.group(1))
            fields.append(
                _field(
                    "expiry_date",
                    m.group(1),
                    text_boxes,
                    date_info["status"],
                    extra={"date_checks": date_info},
                )
            )
            raw_by_name["expiry_date"] = m.group(1)

    if "nationality" not in raw_by_name:
        m = NATIONALITY_RE.search(text)
        if m and "nationality" in profile.expected_fields:
            nat = m.group(1).strip()
            if nat.lower() not in {"of", "the", "and", "for"}:
                fields.append(_field("nationality", nat, text_boxes, "PASS"))
                raw_by_name["nationality"] = nat

    if "sex" not in raw_by_name and "gender" not in raw_by_name:
        m = SEX_RE.search(text)
        if m and ("sex" in profile.expected_fields or "gender" in profile.expected_fields):
            sex_raw = m.group(1).strip().upper()
            sex = {"M": "M", "MALE": "M", "F": "F", "FEMALE": "F"}.get(sex_raw, sex_raw[:1])
            field_key = "gender" if "gender" in profile.expected_fields else "sex"
            fields.append(
                _field(field_key, sex, text_boxes, "PASS" if sex in {"M", "F"} else "WARNING")
            )
            raw_by_name[field_key] = sex

    am = ADDRESS_RE.search(text)
    if am and "address" in profile.expected_fields:
        addr = am.group(1).strip()
        fields.append(_field("address", addr, text_boxes, "WARNING"))
        raw_by_name["address"] = addr

    if document_type == "VISA" or "visa_type" in profile.expected_fields:
        m = VISA_TYPE_RE.search(text)
        if m:
            fields.append(_field("visa_type", m.group(1).strip(), text_boxes, "PASS"))
            raw_by_name["visa_type"] = m.group(1).strip()
        m = ENTRIES_RE.search(text)
        if m:
            fields.append(_field("entries", m.group(1).strip(), text_boxes, "PASS"))
            raw_by_name["entries"] = m.group(1).strip()
        m = DURATION_RE.search(text)
        if m:
            fields.append(
                _field("duration_of_stay", m.group(1).strip(), text_boxes, "PASS")
            )
            raw_by_name["duration_of_stay"] = m.group(1).strip()
        if "visa_number" not in raw_by_name:
            m = VISA_NUMBER_RE.search(text)
            if m:
                fields.append(
                    _field(
                        "visa_number",
                        m.group(1).strip(),
                        text_boxes,
                        _validate_document_number("VISA", m.group(1)),
                    )
                )
                raw_by_name["visa_number"] = m.group(1).strip()

    if document_type == "PERMIT" or "permit_type" in profile.expected_fields:
        m = PERMIT_TYPE_RE.search(text)
        if m:
            ptype = m.group(1).strip()
            if len(ptype) >= 3:
                fields.append(_field("permit_type", ptype, text_boxes, "WARNING"))
                raw_by_name["permit_type"] = ptype

    # Deduplicate by field name (first wins)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for f in fields:
        if f["field"] in seen:
            continue
        seen.add(f["field"])
        deduped.append(f)
    fields = deduped

    applicable = _applicable_ocr_fields(document_type)
    found = {f["field"] for f in fields}
    if "sex" in found and "gender" in applicable:
        found.add("gender")
    if "gender" in found and "sex" in applicable:
        found.add("sex")
    if "visa_number" in found and "document_number" in applicable:
        found.add("document_number")

    # Successfully extracted = EXTRACTED status only (not UNCERTAIN)
    extracted_ok = {
        f["field"] for f in fields if f.get("status") == "EXTRACTED"
    }
    if "visa_number" in extracted_ok and "document_number" in applicable:
        extracted_ok.add("document_number")
    if "sex" in extracted_ok and "gender" in applicable:
        extracted_ok.add("gender")
    if "gender" in extracted_ok and "sex" in applicable:
        extracted_ok.add("sex")

    missing = [f for f in applicable if f not in found]
    uncertain = [f["field"] for f in fields if f.get("status") == "UNCERTAIN"]

    # Full inventory for UI: every applicable field + non-OCR as NOT_APPLICABLE
    inventory: List[Dict[str, Any]] = []
    by_name = {f["field"]: f for f in fields}
    for key in applicable:
        if key in by_name:
            inventory.append(
                {
                    "field": key,
                    "status": by_name[key]["status"],
                    "value": by_name[key].get("value"),
                    "ocr_confidence": by_name[key].get("ocr_confidence"),
                    "validation": by_name[key].get("validation"),
                }
            )
        else:
            inventory.append({"field": key, "status": "MISSING", "value": None})
    for key in profile.expected_fields:
        if key in NON_OCR_EXPECTED:
            inventory.append(
                {
                    "field": key,
                    "status": "NOT_APPLICABLE",
                    "value": None,
                    "note": (
                        "MRZ is assessed only by the MRZ module when supports_mrz=True."
                        if key == "mrz"
                        else "Visual-only field — not counted in OCR completeness."
                    ),
                }
            )

    denom = len(applicable)
    completeness = (
        round(len([f for f in applicable if f in extracted_ok]) / denom, 4)
        if denom
        else (1.0 if document_type in {"UNKNOWN", "OTHER"} and not applicable else 0.0)
    )
    if denom == 0:
        completeness = 0.0

    mean_field_conf = (
        round(sum(f["ocr_confidence"] for f in fields) / len(fields), 4) if fields else None
    )

    consistency = _cross_field_checks(fields, raw_by_name)
    fail_validations = sum(1 for f in fields if f.get("validation") == "FAIL")
    warn_validations = sum(1 for f in fields if f.get("validation") == "WARNING")

    if not fields:
        status = "NO_FIELDS"
    elif completeness >= 0.6 and not uncertain:
        status = "OK"
    else:
        status = "PARTIAL"

    return {
        "status": status,
        "document_type": document_type,
        "profile": profile.profile_id,
        "fields": fields,
        "field_inventory": inventory,
        "applicable_fields": applicable,
        "missing_expected_fields": missing,
        "uncertain_fields": uncertain,
        "field_completeness": completeness,
        "completeness_basis": {
            "applicable_count": denom,
            "extracted_count": len([f for f in applicable if f in extracted_ok]),
            "note": "Completeness = EXTRACTED applicable OCR fields / applicable OCR fields only.",
        },
        "mean_field_ocr_confidence": mean_field_conf,
        "validation_summary": {
            "fail": fail_validations,
            "warning": warn_validations,
            "pass": sum(1 for f in fields if f.get("validation") == "PASS"),
        },
        "consistency_checks": consistency,
        "compare_values": dict(raw_by_name),
        "note": (
            "Field values are masked. Raw PII is not returned. "
            "OCR confidence and field checks are not authenticity scores. "
            "status EXTRACTED/UNCERTAIN/MISSING/NOT_APPLICABLE — never PASS with 0% OCR."
        ),
    }
