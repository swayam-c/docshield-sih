"""SIH Phase 20 — screening report generation (JSON + Markdown).

Reports summarize screening evidence for authorized reviewers.
They do not claim legal authenticity or official government verification.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import get_settings


def build_report_payload(
    *,
    case_id: str,
    document_type: str,
    decision: str,
    assessment: Dict[str, Any],
    evidence: Dict[str, Any],
    screening_result: Optional[Dict[str, Any]] = None,
    upload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cal = evidence.get("calibration") or {}
    conf = evidence.get("confidence_engine") or {}
    dec = evidence.get("decision_engine") or {}
    return {
        "report_type": "DOCSHIELD_SCREENING_SUMMARY",
        "case_id": case_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_type": document_type,
        "decision": decision,
        "authenticity_estimate": assessment.get("authenticity_estimate"),
        "confidence": assessment.get("confidence"),
        "evidence_quality": assessment.get("evidence_quality"),
        "assessment": assessment,
        "evidence_summary": {
            "ocr_status": (evidence.get("ocr") or {}).get("status"),
            "mrz_status": (evidence.get("mrz") or {}).get("status"),
            "fusion_status": (evidence.get("fusion") or {}).get("status"),
            "tampering_severity": (evidence.get("tampering") or {}).get("severity"),
            "face_status": (evidence.get("face") or {}).get("status"),
            "cross_validation": (evidence.get("cross_validation") or {}).get("overall"),
            "calibration": cal.get("status"),
            "confidence_engine": conf.get("overall_confidence"),
            "decision_engine": dec.get("decision"),
        },
        "phase_status": (screening_result or {}).get("phase_status"),
        "upload": upload or {},
        "disclaimer": (
            "This report is an AI-assisted screening summary for authorized reviewers. "
            "It is not a legal authenticity certificate, not an official government "
            "verification result, and not a blockchain-anchored credential unless "
            "explicitly marked otherwise (current build: LOCAL only)."
        ),
        "demo_banner": "DEMO / SYNTHETIC PROXY SIGNALS — NOT OFFICIAL VERIFICATION",
    }


def render_markdown(report: Dict[str, Any]) -> str:
    a = report.get("assessment") or {}
    es = report.get("evidence_summary") or {}
    lines = [
        f"# DOCSHIELD AI Screening Report — {report.get('case_id')}",
        "",
        f"**Generated:** {report.get('generated_at')}",
        f"**Document type:** {report.get('document_type')}",
        f"**Decision:** {report.get('decision')}",
        f"**Authenticity estimate (proxy):** {report.get('authenticity_estimate')}",
        f"**Confidence:** {report.get('confidence')}",
        f"**Evidence quality:** {report.get('evidence_quality')}",
        "",
        "## Evidence summary",
        f"- OCR: {es.get('ocr_status')}",
        f"- MRZ: {es.get('mrz_status')}",
        f"- Fusion: {es.get('fusion_status')}",
        f"- Tampering: {es.get('tampering_severity')}",
        f"- Face: {es.get('face_status')}",
        f"- Cross-validation: {es.get('cross_validation')}",
        f"- Calibration: {es.get('calibration')}",
        "",
        "## Reasons",
    ]
    for r in a.get("reasons") or []:
        lines.append(f"- {r}")
    lines += [
        "",
        f"**Recommended action:** {a.get('recommended_action') or '—'}",
        "",
        "## Disclaimer",
        report.get("disclaimer") or "",
        "",
        f"*{report.get('demo_banner')}*",
        "",
    ]
    return "\n".join(lines)


def save_report(report: Dict[str, Any]) -> Dict[str, str]:
    settings = get_settings()
    out_dir = Path(settings.data_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_id = report["case_id"]
    json_path = out_dir / f"{case_id}.json"
    md_path = out_dir / f"{case_id}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path)}
