"""Public-format OCR field patterns (known shapes only — not secret specs)."""

from __future__ import annotations

import re

PAN_RE = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
AADHAAR_RE = re.compile(r"\b(\d{4}\s?\d{4}\s?\d{4})\b")
PASSPORT_RE = re.compile(r"\b([A-Z][0-9]{7})\b")
DL_RE = re.compile(r"\b([A-Z]{2}[\-\s]?\d{2}[\-\s]?\d{4}[\-\s]?\d{7})\b", re.I)
# Generic alphanumeric IDs (visa/permit/national ID) — loose
GENERIC_ID_RE = re.compile(r"\b([A-Z]{1,3}[0-9]{6,12}[A-Z0-9]?)\b")

NAME_LABEL_RE = re.compile(
    r"(?:(?:surname|given\s*names?|full\s*name|holder(?:'s)?\s*name|"
    r"(?<!father\s)(?<!mother\s)(?<!husband\s)name)\s*[:\-]\s*)"
    r"([A-Z][A-Za-z]+(?:[\s,]+[A-Z][A-Za-z]+){0,4})",
    re.I,
)
FATHER_NAME_RE = re.compile(
    r"(?:father(?:'s)?\s*name|s/?o)\s*[:\-]\s*"
    r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
    re.I,
)
DOB_LABEL_RE = re.compile(
    r"(?:dob|date\s*of\s*birth|birth(?:\s*date)?)\s*[:\-]?\s*"
    r"([0-9]{1,2}[/\-.][0-9]{1,2}[/\-.][0-9]{2,4}|"
    r"[0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4})",
    re.I,
)
DATE_RE = re.compile(
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\b"
)
EXPIRY_RE = re.compile(
    r"(?:expir(?:y|es|ation)|valid\s*(?:till|until|to)|date\s*of\s*expiry)\s*[:\-]?\s*"
    r"([0-9]{1,2}[/\-.][0-9]{1,2}[/\-.][0-9]{2,4}|"
    r"[0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4})",
    re.I,
)
ISSUE_RE = re.compile(
    r"(?:date\s*of\s*issue|issue(?:d)?(?:\s*date)?|doi)\s*[:\-]?\s*"
    r"([0-9]{1,2}[/\-.][0-9]{1,2}[/\-.][0-9]{2,4}|"
    r"[0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4})",
    re.I,
)
NATIONALITY_RE = re.compile(
    r"(?:nationality(?:\s*/\s*nationalit[eé])?|citizen(?:ship)?|country)\s*[:\-]?\s*(?:\n|\r\n)?\s*"
    r"([A-Za-z]{3,20})",
    re.I,
)
SEX_RE = re.compile(
    r"(?:sex(?:\s*/\s*sexe)?|gender)\s*[:\-]?\s*(?:\n|\r\n)?\s*"
    r"([MFmf]|Male|Female|Other|MALE|FEMALE)\b",
    re.I,
)
ADDRESS_RE = re.compile(r"(?:address|addr)\s*[:\-]?\s*(.{8,100})", re.I)

VISA_NUMBER_RE = re.compile(
    r"(?:visa\s*(?:no\.?|number|#))\s*[:\-]?\s*([A-Z0-9]{6,14})",
    re.I,
)
VISA_TYPE_RE = re.compile(
    r"(?:visa\s*type|type\s*of\s*visa|category)\s*[:\-]?\s*([A-Za-z0-9/\-]{2,24})",
    re.I,
)
ENTRIES_RE = re.compile(
    r"(?:entries|number\s*of\s*entries)\s*[:\-]?\s*(Single|Double|Multiple|\d+)",
    re.I,
)
DURATION_RE = re.compile(
    r"(?:duration(?:\s*of\s*stay)?|period\s*of\s*stay|stay)\s*[:\-]?\s*"
    r"(\d+\s*(?:days?|months?|years?)|[A-Za-z0-9 ]{2,20})",
    re.I,
)
PERMIT_TYPE_RE = re.compile(
    r"(?:permit\s*type|type\s*of\s*permit|permit)\s*[:\-]?\s*([A-Za-z0-9/\- ]{3,40})",
    re.I,
)
PASSPORT_LABEL_RE = re.compile(
    r"(?:passport\s*(?:no|number|#))\s*[:\-]?\s*([A-Z][0-9]{7})",
    re.I,
)
PAN_LABEL_RE = re.compile(
    r"(?:permanent\s*account\s*(?:number|no)|pan)\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])",
    re.I,
)
DL_LABEL_RE = re.compile(
    r"(?:(?:dl|licence|license)\s*(?:no|number|#)|driving\s*licen[cs]e)\s*[:\-]?\s*"
    r"([A-Z]{2}[\-\s]?\d{2}[\-\s]?\d{4}[\-\s]?\d{7})",
    re.I,
)
