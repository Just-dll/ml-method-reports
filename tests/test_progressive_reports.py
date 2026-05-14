from __future__ import annotations

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ml_method_reports import EtalonClassifier, report_for
from ml_method_reports.reporting.models import ExperimentReport


def _classification_data():
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=23,
    )
    return train_test_split(X, y, random_state=23, stratify=y)


def _feature_names() -> list[str]:
    return ["feature_a", "feature_b", "feature_c", "feature_d"]


def _section_titles(report: ExperimentReport) -> list[str]:
    return [section.title for section in report.sections]


def _table_contains(report: ExperimentReport, expected: str) -> bool:
    for section in report.sections:
        for row in section.table or []:
            if any(str(value) == expected for value in row.values()):
                return True
    return False


def test_generic_classifier_model_only_report_has_model_summary() -> None:
    X_train, _, y_train, _ = _classification_data()
    model = LogisticRegression(max_iter=300).fit(X_train, y_train)

    report = report_for(model).as_generic().build()

    assert isinstance(report, ExperimentReport)
    assert "1. Model Overview" in _section_titles(report)
    assert _table_contains(report, "LogisticRegression")
    assert _table_contains(report, "generic")


def test_generic_classifier_with_x_test_without_y_test_skips_metrics() -> None:
    X_train, X_test, y_train, _ = _classification_data()
    model = LogisticRegression(max_iter=300).fit(X_train, y_train)

    report = report_for(model).as_generic().with_test_data(
        X_test=X_test,
        feature_names=_feature_names(),
    ).build()

    titles = _section_titles(report)
    assert "6. Prediction Samples" in titles
    assert "Evaluation Availability" in titles
    assert "9. Evaluation Metrics" not in titles
    assert "10. Confusion Matrix" not in titles


def test_generic_classifier_with_x_test_and_y_test_adds_metrics() -> None:
    X_train, X_test, y_train, y_test = _classification_data()
    model = LogisticRegression(max_iter=300).fit(X_train, y_train)

    report = report_for(model).as_generic().with_test_data(
        X_test=X_test,
        y_test=y_test,
        feature_names=_feature_names(),
    ).build()

    titles = _section_titles(report)
    assert "9. Evaluation Metrics" in titles
    assert "10. Confusion Matrix" in titles


def test_knn_model_only_report_does_not_crash() -> None:
    X_train, _, y_train, _ = _classification_data()
    model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

    report = report_for(model).build()

    assert isinstance(report, ExperimentReport)
    assert _table_contains(report, "KNeighborsClassifier")
    assert _table_contains(report, "n_neighbors")


def test_knn_full_data_uses_specialized_report() -> None:
    X_train, X_test, y_train, y_test = _classification_data()
    model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

    report = report_for(model).with_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=_feature_names(),
    ).build()

    titles = _section_titles(report)
    assert report.title == "KNN Classification Report"
    assert "12. Evaluation" in titles
    assert "13. Confusion Matrix" in titles


def test_logistic_regression_fitted_model_only_reports_coefficients() -> None:
    X_train, _, y_train, _ = _classification_data()
    model = LogisticRegression(max_iter=300).fit(X_train, y_train)

    report = report_for(model).build()

    assert "Coefficients" in _section_titles(report)
    assert _table_contains(report, "coef_")


def test_decision_tree_fitted_model_only_reports_tree_artifacts() -> None:
    X_train, _, y_train, _ = _classification_data()
    model = DecisionTreeClassifier(random_state=23).fit(X_train, y_train)

    report = report_for(model).build()

    assert "Feature Importances" in _section_titles(report)
    assert _table_contains(report, "feature_importances_")


def test_kmeans_fitted_model_only_reports_cluster_artifacts() -> None:
    X, _ = make_blobs(n_samples=45, n_features=4, centers=3, random_state=23)
    model = KMeans(n_clusters=3, n_init=5, random_state=23).fit(X)

    report = report_for(model).build()

    assert isinstance(report, ExperimentReport)
    assert _table_contains(report, "cluster_centers_")
    assert _table_contains(report, "labels_")


def test_kmeans_training_data_uses_specialized_report() -> None:
    X, _ = make_blobs(n_samples=45, n_features=4, centers=3, random_state=23)
    model = KMeans(n_clusters=3, n_init=5, random_state=23).fit(X)

    report = report_for(model).with_training_data(X_train=X, feature_names=_feature_names()).build()

    assert report.title == "KMeans Educational Report"
    assert "4. Cluster Centers" in _section_titles(report)


def test_agglomerative_training_data_uses_specialized_report() -> None:
    X, _ = make_blobs(n_samples=45, n_features=4, centers=3, random_state=23)
    model = AgglomerativeClustering(n_clusters=3, compute_distances=True).fit(X)

    report = report_for(model).with_training_data(X_train=X, feature_names=_feature_names()).build()

    assert report.title == "Agglomerative Clustering Educational Report"
    assert "5. Merge Tree Summary" in _section_titles(report)


def test_etalon_classifier_fitted_model_only_reports_centers() -> None:
    X_train, _, y_train, _ = _classification_data()
    model = EtalonClassifier().fit(X_train, y_train)

    report = report_for(model).build()

    assert isinstance(report, ExperimentReport)
    assert _table_contains(report, "EtalonClassifier")
    assert _table_contains(report, "etalon_centers_")


def test_display_model_only_returns_report(monkeypatch) -> None:
    X_train, _, y_train, _ = _classification_data()
    model = LogisticRegression(max_iter=300).fit(X_train, y_train)
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


def test_svc_without_probability_skips_unavailable_probability_section() -> None:
    from sklearn.svm import SVC

    X_train, X_test, y_train, _ = _classification_data()
    model = SVC(probability=False).fit(X_train, y_train)

    report = report_for(model).as_generic().with_test_data(X_test=X_test).build()

    assert isinstance(report, ExperimentReport)
    assert "7. Prediction Probabilities" not in _section_titles(report)
