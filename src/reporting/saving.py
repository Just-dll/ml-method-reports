from __future__ import annotations

import sys
from pathlib import Path

from ml_method_reports.reporting.html_report import HtmlReportGenerator
from ml_method_reports.reporting.models import ExperimentReport
from ml_method_reports.reporting.pdf_report import PdfExperimentReportGenerator


def save_report_bundle(
    report: ExperimentReport,
    output_dir: str | Path,
    *,
    stem: str = "classification_report",
    show_progress: bool = True,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    html_path = output / f"{stem}.html"
    pdf_path = output / f"{stem}.pdf"

    _report_progress(show_progress, f"Saving HTML report to {html_path}...")
    HtmlReportGenerator().save(report, html_path)
    _report_progress(show_progress, "HTML report saved.")

    _report_progress(show_progress, f"Saving PDF report to {pdf_path}...")
    try:
        pdf_path = PdfExperimentReportGenerator().save(report, pdf_path)
    except PermissionError:
        fallback_path = pdf_path.with_name(f"{pdf_path.stem}_latest{pdf_path.suffix}")
        _report_progress(
            show_progress,
            f"PDF file is locked; saving fallback PDF to {fallback_path}...",
        )
        pdf_path = PdfExperimentReportGenerator().save(report, fallback_path)
    _report_progress(show_progress, "PDF report saved.")
    return html_path, pdf_path


def _report_progress(show_progress: bool, message: str) -> None:
    if show_progress:
        print(f"[ml-method-reports] {message}", file=sys.stderr, flush=True)

