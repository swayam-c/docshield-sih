"""PAN card profile — field keys only; no fabricated government specs."""

DOCUMENT_TYPE = "PAN"
PROFILE_ID = "pan"
DISPLAY_NAME = "Permanent Account Number (PAN) Card"
EXPECTED_FIELDS = (
    "name",
    "father_name",
    "date_of_birth",
    "document_number",
    "signature",
)
SUPPORTS_MRZ = False
NOTES = "Use only authorized/reference field patterns during OCR extraction."
