"""Permit document profile — field keys only."""

DOCUMENT_TYPE = "PERMIT"
PROFILE_ID = "permit"
DISPLAY_NAME = "Permit"
EXPECTED_FIELDS = (
    "name",
    "document_number",
    "permit_type",
    "issue_date",
    "expiry_date",
    "nationality",
)
SUPPORTS_MRZ = False
NOTES = "Permit formats vary widely; prefer INCONCLUSIVE when structure is unsupported."
