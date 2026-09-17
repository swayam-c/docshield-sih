"""MRZ detection / parse / check-digit validation (ICAO 9303 style).

Only runs when MRZ-like lines are present. Never fabricates MRZ data.
Compares parsed MRZ to visible OCR fields when both exist.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ocr.dates import parse_date
from ocr.masking import mask_value

# ICAO 9303 character set weights
_MRZ_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
_MRZ_WEIGHTS = (7, 3, 1)


def _char_value(ch: str) -> int:
    ch = ch.upper()
    if ch not in _MRZ_CHARSET:
        return -1
    return _MRZ_CHARSET.index(ch)


def mrz_check_digit(data: str) -> str:
    total = 0
    for i, ch in enumerate(data):
        val = _char_value(ch)
        if val < 0:
            val = 0
        total += val * _MRZ_WEIGHTS[i % 3]
    return str(total % 10)


def verify_check_digit(data: str, digit: str) -> bool:
    if not digit or not str(digit).isdigit():
        return False
    return mrz_check_digit(data) == digit


_MRZ_LINE = re.compile(r"^[A-Z0-9<]{20,44}$")


def extract_mrz_candidates(text: str) -> List[str]:
    """Collect MRZ-like lines. Require '<' fillers to avoid ordinary prose."""
    lines: List[str] = []
    for raw in re.split(r"[\r\n]+", text.upper()):
        cleaned = re.sub(r"[^A-Z0-9<]", "", raw.replace(" ", ""))
        if len(cleaned) >= 20 and cleaned.count("<") >= 2 and _MRZ_LINE.match(cleaned):
            lines.append(cleaned)
    collapsed = re.sub(r"[^A-Z0-9<]", "", text.upper())
    for length in (44, 36, 30):
        for m in re.finditer(rf"[A-Z0-9<]{{{length}}}", collapsed):
            cand = m.group(0)
            if cand.count("<") >= 2:
                lines.append(cand)
    seen = set()
    out: List[str] = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out


def _parse_names_from_line(names_block: str) -> Tuple[str, str]:
    surname, _, rest = names_block.partition("<<")
    given = rest.replace("<", " ").strip()
    surname = surname.replace("<", " ").strip()
    return surname, given


def _parse_td3(line1: str, line2: str) -> Dict[str, Any]:
    """Passport TD3 (2×44)."""
    doc_type = line1[0:2].replace("<", "")
    issuer = line1[2:5].replace("<", "")
    surname, given = _parse_names_from_line(line1[5:44])

    doc_num = line2[0:9]
    doc_cd = line2[9]
    nationality = line2[10:13].replace("<", "")
    dob = line2[13:19]
    dob_cd = line2[19]
    sex = line2[20].replace("<", "")
    expiry = line2[21:27]
    expiry_cd = line2[27]
    optional = line2[28:42]
    optional_cd = line2[42]
    composite_cd = line2[43]

    checks = {
        "document_number": verify_check_digit(doc_num, doc_cd),
        "date_of_birth": verify_check_digit(dob, dob_cd),
        "expiry_date": verify_check_digit(expiry, expiry_cd),
        "optional": verify_check_digit(optional, optional_cd) if optional_cd.isdigit() else None,
    }
    composite_data = doc_num + doc_cd + dob + dob_cd + expiry + expiry_cd + optional + optional_cd
    checks["composite"] = verify_check_digit(composite_data, composite_cd)

    return {
        "format": "TD3",
        "document_type_code": doc_type,
        "issuer": issuer,
        "surname": surname,
        "given_names": given,
        "document_number_raw": doc_num.replace("<", ""),
        "nationality": nationality,
        "date_of_birth_yymmdd": dob,
        "sex": sex,
        "expiry_yymmdd": expiry,
        "check_digits": checks,
        "structure_valid": len(line1) == 44 and len(line2) == 44,
    }


def _parse_td2(line1: str, line2: str) -> Dict[str, Any]:
    """TD2 (2×36) — visas / some IDs."""
    doc_type = line1[0:2].replace("<", "")
    issuer = line1[2:5].replace("<", "")
    surname, given = _parse_names_from_line(line1[5:36])

    doc_num = line2[0:9]
    doc_cd = line2[9]
    nationality = line2[10:13].replace("<", "")
    dob = line2[13:19]
    dob_cd = line2[19]
    sex = line2[20].replace("<", "")
    expiry = line2[21:27]
    expiry_cd = line2[27]
    optional = line2[28:35]
    composite_cd = line2[35]

    checks = {
        "document_number": verify_check_digit(doc_num, doc_cd),
        "date_of_birth": verify_check_digit(dob, dob_cd),
        "expiry_date": verify_check_digit(expiry, expiry_cd),
        "optional": None,
    }
    composite_data = doc_num + doc_cd + dob + dob_cd + expiry + expiry_cd + optional
    checks["composite"] = verify_check_digit(composite_data, composite_cd)

    return {
        "format": "TD2",
        "document_type_code": doc_type,
        "issuer": issuer,
        "surname": surname,
        "given_names": given,
        "document_number_raw": doc_num.replace("<", ""),
        "nationality": nationality,
        "date_of_birth_yymmdd": dob,
        "sex": sex,
        "expiry_yymmdd": expiry,
        "check_digits": checks,
        "structure_valid": len(line1) == 36 and len(line2) == 36,
    }


def _parse_td1(line1: str, line2: str, line3: str) -> Dict[str, Any]:
    """TD1 (3×30) — many national ID cards."""
    doc_type = line1[0:2].replace("<", "")
    issuer = line1[2:5].replace("<", "")
    doc_num = line1[5:14]
    doc_cd = line1[14]
    optional1 = line1[15:30]

    dob = line2[0:6]
    dob_cd = line2[6]
    sex = line2[7].replace("<", "")
    expiry = line2[8:14]
    expiry_cd = line2[14]
    nationality = line2[15:18].replace("<", "")
    optional2 = line2[18:29]
    composite_cd = line2[29]

    surname, given = _parse_names_from_line(line3[0:30])

    checks = {
        "document_number": verify_check_digit(doc_num, doc_cd),
        "date_of_birth": verify_check_digit(dob, dob_cd),
        "expiry_date": verify_check_digit(expiry, expiry_cd),
        "optional": None,
    }
    composite_data = (
        doc_num
        + doc_cd
        + optional1
        + dob
        + dob_cd
        + expiry
        + expiry_cd
        + optional2
    )
    checks["composite"] = verify_check_digit(composite_data, composite_cd)

    return {
        "format": "TD1",
        "document_type_code": doc_type,
        "issuer": issuer,
        "surname": surname,
        "given_names": given,
        "document_number_raw": doc_num.replace("<", ""),
        "nationality": nationality,
        "date_of_birth_yymmdd": dob,
        "sex": sex,
        "expiry_yymmdd": expiry,
        "check_digits": checks,
        "structure_valid": len(line1) == 30 and len(line2) == 30 and len(line3) == 30,
    }


def _select_and_parse(candidates: List[str]) -> Tuple[Optional[Dict[str, Any]], List[str], str]:
    """Return (parsed, lines_used, outcome) outcome in {parsed, partial, none}."""
    td3 = [c for c in candidates if len(c) == 44]
    if len(td3) >= 2:
        # Prefer line starting with P for passport when available
        td3_sorted = sorted(td3, key=lambda x: (0 if x.startswith("P") else 1, -len(x)))
        return _parse_td3(td3_sorted[0], td3_sorted[1]), td3_sorted[:2], "parsed"

    td2 = [c for c in candidates if len(c) == 36]
    if len(td2) >= 2:
        return _parse_td2(td2[0], td2[1]), td2[:2], "parsed"

    td1 = [c for c in candidates if len(c) == 30]
    if len(td1) >= 3:
        return _parse_td1(td1[0], td1[1], td1[2]), td1[:3], "parsed"

    if candidates:
        return None, candidates[:3], "partial"
    return None, [], "none"


def _norm_alpha(s: str) -> str:
    return re.sub(r"[^A-Z]", "", (s or "").upper())


def _yymmdd_from_visible(value: str) -> Optional[str]:
    parsed, status = parse_date(value)
    if status != "PARSED" or parsed is None:
        # try YYMMDD already
        raw = re.sub(r"\D", "", value or "")
        if len(raw) == 6:
            return raw
        return None
    return parsed.strftime("%y%m%d")


def _name_tokens(name: str) -> set:
    parts = re.split(r"[\s,]+", (name or "").upper())
    return {_norm_alpha(t) for t in parts if len(_norm_alpha(t)) >= 2}


def _compare_name(mrz_surname: str, mrz_given: str, visible_name: Optional[str]) -> Dict[str, Any]:
    if not visible_name:
        return {
            "status": "NOT_ASSESSED",
            "detail": "No visible name field to compare.",
        }
    mrz_full = f"{mrz_surname} {mrz_given}".strip()
    vis = _norm_alpha(visible_name)
    mrz_n = _norm_alpha(mrz_full)
    if not vis or not mrz_n:
        return {"status": "NOT_ASSESSED", "detail": "Insufficient name text."}
    if vis == mrz_n or mrz_n in vis or vis in mrz_n:
        return {"status": "PASS", "detail": "Visible name matches MRZ name block."}
    # Token overlap (surname or given present)
    vis_tokens = _name_tokens(visible_name)
    mrz_tokens = _name_tokens(mrz_surname) | _name_tokens(mrz_given)
    if vis_tokens and mrz_tokens and (vis_tokens & mrz_tokens):
        overlap = len(vis_tokens & mrz_tokens) / max(1, len(mrz_tokens))
        if overlap >= 0.5:
            return {"status": "PASS", "detail": "Partial name token overlap with MRZ."}
        return {"status": "WARNING", "detail": "Weak name overlap with MRZ."}
    return {"status": "FAIL", "detail": "Visible name does not match MRZ name."}


def _compare_dob(mrz_yymmdd: str, visible_dob: Optional[str]) -> Dict[str, Any]:
    if not visible_dob:
        return {"status": "NOT_ASSESSED", "detail": "No visible DOB field to compare."}
    vis = _yymmdd_from_visible(visible_dob)
    if not vis or not mrz_yymmdd or len(mrz_yymmdd) != 6:
        return {"status": "NOT_ASSESSED", "detail": "Could not normalize DOB for comparison."}
    if vis == mrz_yymmdd:
        return {"status": "PASS", "detail": "Visible DOB matches MRZ DOB."}
    return {"status": "FAIL", "detail": "Visible DOB does not match MRZ DOB."}


def _compare_doc_number(mrz_doc: str, visible_doc: Optional[str]) -> Dict[str, Any]:
    if not visible_doc:
        return {
            "status": "NOT_ASSESSED",
            "detail": "No visible document number to compare.",
        }
    mrz_n = re.sub(r"[^A-Z0-9]", "", (mrz_doc or "").upper())
    vis_n = re.sub(r"[^A-Z0-9]", "", (visible_doc or "").upper())
    if not mrz_n or not vis_n:
        return {"status": "NOT_ASSESSED", "detail": "Insufficient document number text."}
    if mrz_n == vis_n or mrz_n in vis_n or vis_n in mrz_n:
        return {"status": "PASS", "detail": "Visible document number matches MRZ."}
    return {"status": "FAIL", "detail": "Visible document number does not match MRZ."}


def _visible_map(fields_payload: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Rebuild approximate raw-ish values from masked fields where possible.

    Masked values cannot be compared digit-for-digit for DOB/doc in some cases.
    Prefer unmasked keys if pipeline passes `raw_compare` side channel; else use
    fields listed under optional `compare_values` or fall back to NOT_ASSESSED
    when masking has removed critical digits.
    """
    out: Dict[str, str] = {}
    if not fields_payload:
        return out
    # Preferred: explicit compare_values (unmasked, internal only)
    cv = fields_payload.get("compare_values") or {}
    if isinstance(cv, dict):
        for k, v in cv.items():
            if v:
                out[str(k)] = str(v)
    for f in fields_payload.get("fields") or []:
        name = f.get("field")
        # Use compare_value if present on field
        if name and f.get("compare_value"):
            out[name] = str(f["compare_value"])
    return out


def build_checklist(
    *,
    structure_valid: Optional[bool],
    check_digits_ok: Optional[bool],
    name_cmp: Dict[str, Any],
    dob_cmp: Dict[str, Any],
    doc_cmp: Dict[str, Any],
) -> List[Dict[str, Any]]:
    def item(key: str, label: str, status: str, detail: str = "") -> Dict[str, Any]:
        return {"item": key, "label": label, "status": status, "detail": detail}

    fmt_status = (
        "NOT_ASSESSED"
        if structure_valid is None
        else ("PASS" if structure_valid else "FAIL")
    )
    cd_status = (
        "NOT_ASSESSED"
        if check_digits_ok is None
        else ("PASS" if check_digits_ok else "FAIL")
    )
    return [
        item("format_valid", "Format valid", fmt_status),
        item("check_digits_valid", "Check digits valid", cd_status),
        item("name_consistency", "Name consistency", name_cmp["status"], name_cmp.get("detail", "")),
        item("dob_consistency", "DOB consistency", dob_cmp["status"], dob_cmp.get("detail", "")),
        item(
            "document_number_consistency",
            "Document number consistency",
            doc_cmp["status"],
            doc_cmp.get("detail", ""),
        ),
    ]


def _mask_mrz_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    masked = {}
    for k, v in fields.items():
        if v is None:
            masked[k] = None
            continue
        if k in {"surname", "given_names", "document_number"}:
            masked[k] = mask_value("name" if "name" in k or k != "document_number" else "document_number", str(v))
            if k == "document_number":
                masked[k] = mask_value("document_number", str(v))
            elif k in {"surname", "given_names"}:
                masked[k] = mask_value("name", str(v))
        elif k in {"date_of_birth_yymmdd", "expiry_yymmdd"}:
            masked[k] = re.sub(r"\d", "X", str(v))
        else:
            masked[k] = v
    return masked


def analyze_mrz(
    full_text: str,
    document_type: str,
    supports_mrz: bool,
    fields_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    empty_cmp = {"status": "NOT_ASSESSED", "detail": "MRZ not assessed."}
    if not supports_mrz:
        return {
            "status": "NOT_APPLICABLE",
            "message": "MRZ not applicable for this document type.",
            "checklist": [
                {
                    "item": "mrz_applicable",
                    "label": "MRZ applicable",
                    "status": "NOT_APPLICABLE",
                    "detail": "This document type does not use an MRZ.",
                }
            ],
            "format_valid": "NOT_APPLICABLE",
            "check_digits_valid": "NOT_APPLICABLE",
            "name_consistency": "NOT_APPLICABLE",
            "dob_consistency": "NOT_APPLICABLE",
            "document_number_consistency": "NOT_APPLICABLE",
        }

    candidates = extract_mrz_candidates(full_text or "")
    if not candidates:
        checklist = build_checklist(
            structure_valid=None,
            check_digits_ok=None,
            name_cmp=empty_cmp,
            dob_cmp=empty_cmp,
            doc_cmp=empty_cmp,
        )
        for row in checklist:
            row["status"] = "NOT_ASSESSED"
        return {
            "status": "NOT_DETECTED",
            "message": "No MRZ-like lines detected. Never fabricated.",
            "lines": [],
            "checklist": checklist,
        }

    parsed, lines_used, outcome = _select_and_parse(candidates)
    if outcome == "partial" or parsed is None:
        checklist = build_checklist(
            structure_valid=False,
            check_digits_ok=None,
            name_cmp=empty_cmp,
            dob_cmp=empty_cmp,
            doc_cmp=empty_cmp,
        )
        return {
            "status": "PARTIAL",
            "message": "MRZ-like text found but structure incomplete for TD1/TD2/TD3 parse.",
            "lines_masked": ["X" * min(44, len(c)) for c in lines_used],
            "line_lengths": [len(c) for c in lines_used],
            "checklist": checklist,
            "note": "Incomplete MRZ ≠ fabricated result.",
        }

    check_ok = all(v for v in parsed["check_digits"].values() if v is not None)
    visible = _visible_map(fields_payload)
    # Prefer unmasked compare_values from field extraction
    name_cmp = _compare_name(
        parsed["surname"], parsed["given_names"], visible.get("name")
    )
    dob_cmp = _compare_dob(parsed["date_of_birth_yymmdd"], visible.get("date_of_birth"))
    doc_cmp = _compare_doc_number(
        parsed["document_number_raw"],
        visible.get("document_number")
        or visible.get("visa_number")
        or visible.get("passport_number"),
    )

    checklist = build_checklist(
        structure_valid=bool(parsed["structure_valid"]),
        check_digits_ok=bool(check_ok),
        name_cmp=name_cmp,
        dob_cmp=dob_cmp,
        doc_cmp=doc_cmp,
    )

    fields_out = _mask_mrz_fields(
        {
            "surname": parsed["surname"],
            "given_names": parsed["given_names"],
            "document_number": parsed["document_number_raw"],
            "nationality": parsed["nationality"],
            "date_of_birth_yymmdd": parsed["date_of_birth_yymmdd"],
            "expiry_yymmdd": parsed["expiry_yymmdd"],
            "sex": parsed["sex"],
            "issuer": parsed["issuer"],
        }
    )

    return {
        "status": "PARSED",
        "structure": parsed["format"],
        "structure_valid": parsed["structure_valid"],
        "check_digits": parsed["check_digits"],
        "check_digits_ok": check_ok,
        "fields": fields_out,
        "consistency": {
            "name": name_cmp,
            "date_of_birth": dob_cmp,
            "document_number": doc_cmp,
        },
        "checklist": checklist,
        "lines_count": len(lines_used),
        "note": (
            "MRZ values are masked. Check-digit or consistency FAIL ≠ automatic fraud; "
            "use as a review signal only."
        ),
    }
