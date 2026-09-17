"""Tamper-evident local audit hash chain (SIH Phase 21–22)."""

from audit.chain import (
    append_audit_event,
    chain_status,
    list_case_audit,
    verify_chain,
)

__all__ = [
    "append_audit_event",
    "chain_status",
    "list_case_audit",
    "verify_chain",
]
