from __future__ import annotations

from pathlib import Path

from ml_method_reports.reporting.html_report import HtmlReportGenerator
from ml_method_reports.reporting.models import ExperimentReport

NOTEBOOK_EXTRA_ERROR = (
    "Notebook display requires the notebook extra:\n"
    'pip install "ml-method-reports[notebook]"'
)


def render_notebook_html(report: ExperimentReport, base_dir: str | Path | None = None) -> str:
    return HtmlReportGenerator().render(report, embed_images=True, base_dir=base_dir)


def display_report(report: ExperimentReport, base_dir: str | Path | None = None) -> None:
    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise RuntimeError(NOTEBOOK_EXTRA_ERROR) from exc

    display(HTML(render_notebook_html(report, base_dir=base_dir)))
