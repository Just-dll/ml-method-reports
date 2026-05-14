from __future__ import annotations

import sys
from pathlib import Path

from ml_method_reports.reporting.html_report import HtmlReportGenerator
from ml_method_reports.reporting.models import ExperimentReport
from ml_method_reports.reporting.pdf_report import PdfExperimentReportGenerator


def save_html_report(
    report: ExperimentReport,
    output_path: str | Path,
    *,
    show_progress: bool = True,
) -> Path:
    path = Path(output_path)
    _report_progress(show_progress, f"Saving HTML report to {path}...")
    path = HtmlReportGenerator().save(report, path)
    _report_progress(show_progress, "HTML report saved.")
    return path


def save_pdf_report(
    report: ExperimentReport,
    output_path: str | Path,
    *,
    show_progress: bool = True,
) -> Path:
    path = Path(output_path)
    _report_progress(show_progress, f"Saving PDF report to {path}...")
    try:
        path = PdfExperimentReportGenerator().save(report, path)
    except PermissionError:
        fallback_path = path.with_name(f"{path.stem}_latest{path.suffix}")
        _report_progress(
            show_progress,
            f"PDF file is locked; saving fallback PDF to {fallback_path}...",
        )
        path = PdfExperimentReportGenerator().save(report, fallback_path)
    _report_progress(show_progress, "PDF report saved.")
    return path


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

    html_path = save_html_report(report, html_path, show_progress=show_progress)
    pdf_path = save_pdf_report(report, pdf_path, show_progress=show_progress)
    return html_path, pdf_path


def _report_progress(show_progress: bool, message: str) -> None:
    if show_progress:
        print(f"[ml-method-reports] {message}", file=sys.stderr, flush=True)

