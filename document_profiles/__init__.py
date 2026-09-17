"""Document-type field profiles (SIH 26188 taxonomy + Indian subtypes).

Only known/authorized reference field keys — not invented government specs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from document_profiles import (
    aadhaar,
    driving_licence,
    generic,
    national_id,
    pan,
    passport,
    permit,
    unknown,
    visa,
)

DocumentType = str

# SIH core screening classes
SIH_CORE_TYPES: Tuple[str, ...] = (
    "PASSPORT",
    "VISA",
    "NATIONAL_ID",
    "DRIVING_LICENCE",
    "PERMIT",
    "OTHER",
    "UNKNOWN",
)

# Full label set including Indian fine-grained subtypes
DOCUMENT_TYPES: Tuple[str, ...] = (
    "PASSPORT",
    "VISA",
    "NATIONAL_ID",
    "DRIVING_LICENCE",
    "PERMIT",
    "PAN",
    "AADHAAR",
    "OTHER",
    "UNKNOWN",
)

# Map subtypes → SIH category
SIH_CATEGORY: Dict[str, str] = {
    "PASSPORT": "PASSPORT",
    "VISA": "VISA",
    "NATIONAL_ID": "NATIONAL_ID",
    "DRIVING_LICENCE": "DRIVING_LICENCE",
    "PERMIT": "PERMIT",
    "PAN": "NATIONAL_ID",
    "AADHAAR": "NATIONAL_ID",
    "OTHER": "OTHER",
    "UNKNOWN": "UNKNOWN",
}


@dataclass(frozen=True)
class DocumentProfile:
    profile_id: str
    document_type: str
    display_name: str
    expected_fields: Tuple[str, ...]
    supports_mrz: bool = False
    notes: str = ""


def _profile_from_module(mod) -> DocumentProfile:
    return DocumentProfile(
        profile_id=getattr(mod, "PROFILE_ID", mod.DOCUMENT_TYPE.lower()),
        document_type=mod.DOCUMENT_TYPE,
        display_name=getattr(mod, "DISPLAY_NAME", mod.DOCUMENT_TYPE),
        expected_fields=tuple(getattr(mod, "EXPECTED_FIELDS", ())),
        supports_mrz=bool(getattr(mod, "SUPPORTS_MRZ", False)),
        notes=getattr(mod, "NOTES", ""),
    )


PROFILE_BY_TYPE: Dict[str, DocumentProfile] = {
    passport.DOCUMENT_TYPE: _profile_from_module(passport),
    visa.DOCUMENT_TYPE: _profile_from_module(visa),
    national_id.DOCUMENT_TYPE: _profile_from_module(national_id),
    driving_licence.DOCUMENT_TYPE: _profile_from_module(driving_licence),
    permit.DOCUMENT_TYPE: _profile_from_module(permit),
    pan.DOCUMENT_TYPE: _profile_from_module(pan),
    aadhaar.DOCUMENT_TYPE: _profile_from_module(aadhaar),
    generic.DOCUMENT_TYPE: _profile_from_module(generic),
    unknown.DOCUMENT_TYPE: _profile_from_module(unknown),
}


def get_profile(document_type: str) -> DocumentProfile:
    return PROFILE_BY_TYPE.get(document_type, PROFILE_BY_TYPE["UNKNOWN"])


def list_profiles() -> List[DocumentProfile]:
    return [PROFILE_BY_TYPE[t] for t in DOCUMENT_TYPES if t in PROFILE_BY_TYPE]


def to_sih_category(label: str) -> str:
    return SIH_CATEGORY.get(label, "UNKNOWN")
