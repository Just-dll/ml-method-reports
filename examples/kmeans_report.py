from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_example_utils import log_saved_report, standard_scale_frames  # noqa: E402

from ml_method_reports.reporting import report_for  # noqa: E402

FEATURE_NAMES = ["feature_a", "feature_b", "feature_c", "feature_d"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a KMeans educational report.")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runtime" / "reports" / "kmeans_example"
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    html_path, pdf_path = create_report_example(
        output_dir=args.output_dir,
        random_state=args.random_state,
    )
    log_saved_report(html_path, pdf_path)


def create_report_example(*, output_dir: Path | str, random_state: int = 42) -> tuple[Path, Path]:
    X, _ = make_blobs(
        n_samples=160, n_features=len(FEATURE_NAMES), centers=3, random_state=random_state
    )
    dataframe = pd.DataFrame(X, columns=FEATURE_NAMES)
    X_scaled, _, scaling_params = standard_scale_frames(dataframe, dataframe)
    model = KMeans(n_clusters=3, random_state=random_state, n_init="auto").fit(X_scaled)
    return (
        report_for(model)
        .with_test_data(X_test=X_scaled, feature_names=FEATURE_NAMES)
        .with_options(
            dataset_source="sklearn.datasets.make_blobs",
            scaling_method="standard",
            scaling_params=scaling_params,
        )
        .save(output_dir, stem="kmeans_report")
    )


if __name__ == "__main__":
    main()
