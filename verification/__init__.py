"""Visual forensics / identity verification packages.

Face detection lives in verification.face (SIH Phase 11).
Unavailable official providers must return UNAVAILABLE — never fabricate.
"""

from verification.camera import (
    analyze_camera_frame,
    attach_document_reference,
    camera_module_status,
    end_camera_session,
    start_camera_session,
)
from verification.face import analyze_document_face, compare_faces, detect_faces
from verification.liveness import assess_passive_liveness

__all__ = [
    "analyze_document_face",
    "compare_faces",
    "detect_faces",
    "analyze_camera_frame",
    "attach_document_reference",
    "camera_module_status",
    "end_camera_session",
    "start_camera_session",
    "assess_passive_liveness",
]
