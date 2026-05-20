"""Deterministic validation IDs for manual validation cohort exports."""

from __future__ import annotations

from typing import List, Sequence


def format_validation_patient_id(patient_index: int) -> str:
    """1-based index → ``Patient_0001``."""
    return f"Patient_{patient_index:04d}"


def format_validation_report_id(validation_patient_id: str, report_nr_within_patient: int) -> str:
    """``Patient_0001_Report_0002``."""
    return f"{validation_patient_id}_Report_{report_nr_within_patient:04d}"


def assign_validation_patient_ids(sorted_patient_ids: Sequence[str]) -> dict[str, str]:
    """Map ``PatientenID`` → ``validation_patient_id`` (stable given sorted input order)."""
    return {
        pid: format_validation_patient_id(i + 1)
        for i, pid in enumerate(sorted_patient_ids)
    }
