"""Visa document profile — field keys only; no fabricated government specs."""

DOCUMENT_TYPE = "VISA"
PROFILE_ID = "visa"
DISPLAY_NAME = "Visa"
EXPECTED_FIELDS = (
    "name",
    "visa_number",
    "visa_type",
    "nationality",
    "date_of_birth",
    "issue_date",
    "expiry_date",
    "entries",
    "duration_of_stay",
)
SUPPORTS_MRZ = True
NOTES = "Visa layouts vary; MRZ parsed only when machine-readable lines are actually detected."
