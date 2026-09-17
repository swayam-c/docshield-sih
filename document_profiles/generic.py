"""Generic / OTHER document profile fallback."""

DOCUMENT_TYPE = "OTHER"
PROFILE_ID = "generic"
DISPLAY_NAME = "Other / Unsupported Document"
EXPECTED_FIELDS = (
    "name",
    "document_number",
    "date_of_birth",
)
SUPPORTS_MRZ = False
NOTES = "Unsupported types should prefer INCONCLUSIVE when critical evidence is missing."
