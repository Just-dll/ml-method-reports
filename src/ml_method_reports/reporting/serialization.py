from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ml_method_reports.reporting.types import ReportValue, TableRows


def to_report_value(value: object, *, float_digits: int = 6) -> ReportValue:
    """Convert numpy/scientific values into stable report-friendly Python values."""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), float_digits)
    if isinstance(value, float):
        return round(value, float_digits)
    if isinstance(value, np.ndarray):
        return to_report_value(value.tolist(), float_digits=float_digits)
    if isinstance(value, Mapping):
        return {
            to_report_value(key, float_digits=float_digits): to_report_value(
                item,
                float_digits=float_digits,
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [to_report_value(item, float_digits=float_digits) for item in value]
    return value


def sanitize_table_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    float_digits: int = 6,
) -> TableRows:
    return [
        {
            str(to_report_value(key, float_digits=float_digits)): to_report_value(
                value,
                float_digits=float_digits,
            )
            for key, value in row.items()
        }
        for row in rows
    ]


def format_report_value(value: object) -> str:
    sanitized = to_report_value(value)
    if isinstance(sanitized, float):
        return f"{sanitized:.6g}"
    if isinstance(sanitized, list):
        return "[" + ", ".join(format_report_value(item) for item in sanitized) + "]"
    if isinstance(sanitized, dict):
        items = [
            f"{format_report_value(key)}: {format_report_value(item)}"
            for key, item in sanitized.items()
        ]
        return "{" + ", ".join(items) + "}"
    return str(sanitized)

