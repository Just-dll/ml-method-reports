from __future__ import annotations

import base64
import builtins
import types

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ml_method_reports import report_for
from ml_method_reports.reporting.html_report import HtmlReportGenerator
from ml_method_reports.reporting.models import ExperimentReport, ReportSection


def _model() -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=30,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    model = LogisticRegression(max_iter=200).fit(X, y)
    return model, X, y


def test_build_works_without_notebook_dependencies() -> None:
    model, _, _ = _model()

    report = report_for(model).build()

    assert isinstance(report, ExperimentReport)
    assert "LogisticRegression" in report.title


def test_save_still_generates_html_and_pdf(tmp_path) -> None:
    model, X, y = _model()

    html_path, pdf_path = report_for(model).with_test_data(X_test=X, y_test=y).save(tmp_path)

    assert html_path.exists()
    assert pdf_path.exists()


def test_save_html_generates_only_html_report(tmp_path) -> None:
    model, X, y = _model()
    html_path = tmp_path / "single.html"

    saved_path = report_for(model).with_test_data(X_test=X, y_test=y).save_html(html_path)

    assert saved_path == html_path
    assert html_path.exists()
    assert not (tmp_path / "single.pdf").exists()


def test_save_pdf_generates_only_pdf_report(tmp_path) -> None:
    model, X, y = _model()
    pdf_path = tmp_path / "single.pdf"

    saved_path = report_for(model).with_test_data(X_test=X, y_test=y).save_pdf(pdf_path)

    assert saved_path == pdf_path
    assert pdf_path.exists()
    assert not (tmp_path / "single.html").exists()


def test_render_embed_images_uses_base64_data_uri(tmp_path) -> None:
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGA"
        "WjR9awAAAABJRU5ErkJggg=="
    )
    image_path = tmp_path / "plot.png"
    image_path.write_bytes(image_bytes)
    report = ExperimentReport(
        title="Image report",
        sections=[ReportSection(title="Plot", image_path=image_path)],
    )

    html = HtmlReportGenerator().render(report, embed_images=True)

    assert 'src="data:image/png;base64,' in html


def test_missing_image_path_does_not_crash_render() -> None:
    report = ExperimentReport(
        title="Missing image report",
        sections=[ReportSection(title="Plot", image_path="missing.png")],
    )

    html = HtmlReportGenerator().render(report, embed_images=True)

    assert "Image unavailable: missing.png" in html


def test_display_raises_clear_error_when_ipython_missing(monkeypatch) -> None:
    from ml_method_reports.reporting import notebook

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "IPython.display":
            raise ImportError("no IPython")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="Notebook display requires the notebook extra"):
        notebook.display_report(ExperimentReport(title="Demo"))


def test_display_uses_notebook_renderer_when_ipython_available(monkeypatch) -> None:
    from ml_method_reports.reporting import notebook

    calls: dict[str, object] = {}
    fake_display_module = types.ModuleType("IPython.display")

    class FakeHTML:
        def __init__(self, html: str) -> None:
            calls["html"] = html

    def fake_display(value: object) -> None:
        calls["displayed"] = value

    fake_display_module.HTML = FakeHTML
    fake_display_module.display = fake_display
    fake_ipython = types.ModuleType("IPython")
    fake_ipython.display = fake_display_module
    monkeypatch.setitem(__import__("sys").modules, "IPython", fake_ipython)
    monkeypatch.setitem(__import__("sys").modules, "IPython.display", fake_display_module)

    notebook.display_report(ExperimentReport(title="Inline demo"))

    assert "Inline demo" in str(calls["html"])
    assert isinstance(calls["displayed"], FakeHTML)


def test_build_with_x_test_without_y_test_does_not_crash() -> None:
    model, X, _ = _model()

    report = report_for(model).with_test_data(X_test=X).build()

    titles = [section.title for section in report.sections]
    assert "6. Prediction Samples" in titles
    assert "9. Evaluation Metrics" not in titles


def test_display_model_only_returns_report(monkeypatch) -> None:
    model, _, _ = _model()
    displayed: dict[str, ExperimentReport] = {}

    def fake_display_report(report: ExperimentReport, base_dir=None) -> None:
        displayed["report"] = report
        displayed["base_dir"] = base_dir

    import ml_method_reports.reporting.notebook as notebook

    monkeypatch.setattr(notebook, "display_report", fake_display_report)

    report = report_for(model).display()

    assert report is displayed["report"]
    assert displayed["base_dir"] is not None
    assert isinstance(report, ExperimentReport)
