from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import ml_method_reports
from ml_method_reports import EtalonClassifier, report_for
from ml_method_reports.reporting import ExperimentReport, ReportSection


def _dataset():
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=11,
    )
    return train_test_split(X, y, random_state=11, stratify=y)


def test_public_imports_work() -> None:
    assert ml_method_reports.report_for is report_for
    assert EtalonClassifier.__name__ == "EtalonClassifier"
    assert ExperimentReport.__name__ == "ExperimentReport"
    assert ReportSection.__name__ == "ReportSection"


def test_etalon_classifier_fit_predict_work() -> None:
    X_train, X_test, y_train, _ = _dataset()
    model = EtalonClassifier().fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert predictions.shape[0] == X_test.shape[0]
    assert set(predictions).issubset(set(model.classes_))


def test_report_for_build_and_save_work(tmp_path) -> None:
    X_train, X_test, y_train, y_test = _dataset()
    model = EtalonClassifier().fit(X_train, y_train)

    request = report_for(model).with_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=["f1", "f2", "f3", "f4"],
    )
    report = request.build()
    html_path, pdf_path = request.save(tmp_path, stem="smoke")

    assert isinstance(report, ExperimentReport)
    assert html_path.exists()
    assert pdf_path.exists()


def test_example_catalog_module_imports() -> None:
    examples_path = Path(__file__).resolve().parents[1] / "examples"
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))

    import generate_all_reports

    assert generate_all_reports.REPORTS


def test_package_metadata_references_existing_files() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["name"] == "ml-method-reports"
    assert (root / project["readme"]).exists()
    assert project["license"] == "MIT"
    assert all((root / path).exists() for path in project["license-files"])
    assert (root / "src" / "ml_method_reports" / "__init__.py").exists()
