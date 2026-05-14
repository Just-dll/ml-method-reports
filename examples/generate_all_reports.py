from __future__ import annotations

import html
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from agglomerative_report import create_report_example as create_agglomerative_report  # noqa: E402
from classification_report import create_report_example as create_etalon_report  # noqa: E402
from decision_tree_report import create_report_example as create_decision_tree_report  # noqa: E402
from kmeans_report import create_report_example as create_kmeans_report  # noqa: E402
from knn_classification_report import create_report_example as create_knn_report  # noqa: E402
from logistic_regression_report import create_report_example as create_logistic_report  # noqa: E402
from random_forest_report import create_report_example as create_random_forest_report  # noqa: E402
from svc_report import create_report_example as create_svc_report  # noqa: E402


@dataclass(frozen=True)
class DemoReport:
    name: str
    report_type: str
    explanation: str
    output_slug: str
    factory: Callable[..., tuple[Path, Path]]


REPORTS = [
    DemoReport(
        "Etalon",
        "Custom classifier",
        "Class etalons, distances, and selected prediction.",
        "etalon",
        create_etalon_report,
    ),
    DemoReport(
        "KNN",
        "sklearn classifier",
        "Nearest neighbors, distances, and voting.",
        "knn",
        create_knn_report,
    ),
    DemoReport(
        "Logistic Regression",
        "sklearn classifier",
        "Coefficients, probabilities, and feature influence.",
        "logistic_regression",
        create_logistic_report,
    ),
    DemoReport(
        "Decision Tree",
        "sklearn classifier",
        "Rules, decision path, and feature importance.",
        "decision_tree",
        create_decision_tree_report,
    ),
    DemoReport(
        "Random Forest",
        "sklearn classifier",
        "Ensemble summary, feature importance, and errors.",
        "random_forest",
        create_random_forest_report,
    ),
    DemoReport(
        "SVC",
        "sklearn classifier",
        "Support vectors and decision scores.",
        "svc",
        create_svc_report,
    ),
    DemoReport(
        "KMeans",
        "sklearn clustering",
        "Cluster centers, distances, and inertia.",
        "kmeans",
        create_kmeans_report,
    ),
    DemoReport(
        "Agglomerative",
        "sklearn clustering",
        "Merge tree, cluster sizes, and PCA projection.",
        "agglomerative",
        create_agglomerative_report,
    ),
]


def main() -> None:
    output_root = PROJECT_ROOT / "runtime" / "reports"
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[DemoReport, Path, Path]] = []
    for demo in REPORTS:
        report_dir = output_root / demo.output_slug
        html_path, pdf_path = demo.factory(output_dir=report_dir)
        rows.append((demo, html_path, pdf_path))

    index_path = output_root / "index.html"
    index_path.write_text(_render_catalog(rows, output_root), encoding="utf-8")
    print(f"Catalog saved to {index_path}")


def _render_catalog(rows: list[tuple[DemoReport, Path, Path]], output_root: Path) -> str:
    body_rows = "\n".join(_render_row(demo, html_path, pdf_path, output_root) for demo, html_path, pdf_path in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ml-method-reports demo catalog</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 0.65rem; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    a {{ color: #0969da; }}
  </style>
</head>
<body>
  <h1>ml-method-reports demo catalog</h1>
  <table>
    <thead>
      <tr>
        <th>Report</th>
        <th>Type</th>
        <th>What it explains</th>
        <th>HTML</th>
        <th>PDF</th>
        <th>Notebook</th>
      </tr>
    </thead>
    <tbody>
{body_rows}
    </tbody>
  </table>
</body>
</html>
"""


def _render_row(demo: DemoReport, html_path: Path, pdf_path: Path, output_root: Path) -> str:
    html_link = html.escape(_relative_link(html_path, output_root))
    pdf_link = html.escape(_relative_link(pdf_path, output_root))
    notebook_link = "../../notebooks/00_colab_quickstart.ipynb"
    return f"""      <tr>
        <td>{html.escape(demo.name)}</td>
        <td>{html.escape(demo.report_type)}</td>
        <td>{html.escape(demo.explanation)}</td>
        <td><a href="{html_link}">HTML</a></td>
        <td><a href="{pdf_link}">PDF</a></td>
        <td><a href="{notebook_link}">Quickstart</a></td>
      </tr>"""


def _relative_link(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


if __name__ == "__main__":
    main()
