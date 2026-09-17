"""Driving licence profile — field keys only."""

DOCUMENT_TYPE = "DRIVING_LICENCE"
PROFILE_ID = "driving_licence"
DISPLAY_NAME = "Driving Licence"
EXPECTED_FIELDS = (
    "name",
    "date_of_birth",
    "document_number",
    "issue_date",
    "expiry_date",
    "address",
    "photo",
)
SUPPORTS_MRZ = False
NOTES = "Field layout varies by issuing authority; treat mismatches as review signals only."
