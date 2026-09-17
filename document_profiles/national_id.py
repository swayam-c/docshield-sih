"""National ID profile (generic) — covers Aadhaar-like and other national IDs."""

DOCUMENT_TYPE = "NATIONAL_ID"
PROFILE_ID = "national_id"
DISPLAY_NAME = "National ID"
EXPECTED_FIELDS = (
    "name",
    "date_of_birth",
    "gender",
    "document_number",
    "address",
    "photo",
)
SUPPORTS_MRZ = True
NOTES = (
    "Use jurisdiction-specific subtypes (e.g. AADHAAR/PAN) when cues are strong. "
    "MRZ (TD1) assessed only when machine-readable lines are detected."
)
