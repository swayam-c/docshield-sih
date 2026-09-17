"""Passport profile — field keys only; MRZ applicable where present."""

DOCUMENT_TYPE = "PASSPORT"
PROFILE_ID = "passport"
DISPLAY_NAME = "Passport"
EXPECTED_FIELDS = (
    "name",
    "date_of_birth",
    "document_number",
    "nationality",
    "sex",
    "issue_date",
    "expiry_date",
    "mrz",
    "photo",
)
SUPPORTS_MRZ = True
NOTES = "MRZ structure/check digits validated only when MRZ text is actually detected."
