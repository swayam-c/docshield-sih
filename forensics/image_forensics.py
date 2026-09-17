"""Lightweight image forensics — re-exports SIH Phase 9 tampering module."""

from forensics.tampering import analyze_image_forensics, analyze_tampering

__all__ = ["analyze_image_forensics", "analyze_tampering"]
