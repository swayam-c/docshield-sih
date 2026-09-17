"""Aadhaar profile — field keys only; no fabricated government specs."""

DOCUMENT_TYPE = "AADHAAR"
PROFILE_ID = "aadhaar"
DISPLAY_NAME = "Aadhaar Card"
EXPECTED_FIELDS = (
    "name",
    "date_of_birth",
    "gender",
    "document_number",
    "address",
    "photo",
)
SUPPORTS_MRZ = False
NOTES = "Mask Aadhaar numbers in UI/logs. Prefer authorized verification sources when available."
