"""Date field format and plausibility checks.

Expired ≠ fake. Impossible dates → validation FAIL/WARNING.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional, Tuple


DATE_PATTERNS = [
    (re.compile(r"^(\d{2})[/\-.](\d{2})[/\-.](\d{4})$"), "%d/%m/%Y"),
    (re.compile(r"^(\d{4})[/\-.](\d{2})[/\-.](\d{2})$"), "%Y/%m/%d"),
    (re.compile(r"^(\d{2})[/\-.](\d{2})[/\-.](\d{2})$"), "%d/%m/%y"),
    (re.compile(r"^(\d{2})\s([A-Za-z]{3})\s(\d{4})$"), "%d %b %Y"),
]


def parse_date(value: str) -> Tuple[Optional[date], str]:
    raw = (value or "").strip()
    if not raw:
        return None, "EMPTY"
    for cre, fmt in DATE_PATTERNS:
        m = cre.match(raw)
        if not m:
            continue
        candidate = raw
        if fmt == "%d/%m/%Y":
            candidate = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        elif fmt == "%Y/%m/%d":
            candidate = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        elif fmt == "%d/%m/%y":
            candidate = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        try:
            if fmt == "%d/%m/%Y":
                dt = datetime.strptime(candidate, "%d/%m/%Y").date()
            elif fmt == "%Y/%m/%d":
                dt = datetime.strptime(
                    f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%m-%d"
                ).date()
            elif fmt == "%d/%m/%y":
                dt = datetime.strptime(candidate, "%d/%m/%y").date()
            else:
                dt = datetime.strptime(raw, "%d %b %Y").date()
            return dt, "PARSED"
        except ValueError:
            return None, "IMPOSSIBLE_DATE"
    return None, "UNRECOGNIZED_FORMAT"


def validate_date_field(field: str, value: str, today: Optional[date] = None) -> dict:
    today = today or date.today()
    parsed, status = parse_date(value)
    result = {
        "field": field,
        "status": "PASS",
        "parsed_iso": None,
        "flags": [],
        "note": None,
    }
    if status == "EMPTY":
        result["status"] = "FAIL"
        result["flags"].append("empty")
        return result
    if status == "UNRECOGNIZED_FORMAT":
        result["status"] = "WARNING"
        result["flags"].append("unrecognized_format")
        return result
    if status == "IMPOSSIBLE_DATE" or parsed is None:
        result["status"] = "FAIL"
        result["flags"].append("impossible_date")
        return result

    result["parsed_iso"] = parsed.isoformat()
    if parsed > today.replace(year=today.year + 30):
        result["status"] = "WARNING"
        result["flags"].append("far_future")
    if field == "date_of_birth":
        if parsed > today:
            result["status"] = "FAIL"
            result["flags"].append("dob_in_future")
        age = (today - parsed).days / 365.25
        if age > 120:
            result["status"] = "WARNING"
            result["flags"].append("implausible_age")
    if field == "expiry_date":
        if parsed < today:
            result["flags"].append("expired")
            result["note"] = (
                "Document appears EXPIRED. Expiry does not determine authenticity."
            )
            # Expired ≠ fake — keep PASS on format, flag expired
            result["status"] = "PASS"
    if field == "issue_date" and parsed > today:
        result["status"] = "FAIL"
        result["flags"].append("issue_in_future")
    return result
