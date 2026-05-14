from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from report_example_utils import (  # noqa: E402
    DATASET_SOURCE,
    TARGET_NAME,
    build_classification_dataset,
    log_saved_report,
)

from ml_method_reports import EtalonClassifier  # noqa: E402
from ml_method_reports.reporting import report_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an educational method-of-etalons classification report."
    )
    parser.add_argument("--feature-index", type=int, default=0)
    parser.add_argument(
        "--metric",
        choices=["euclidean", "manhattan", "chebyshev", "cosine"],
        default="euclidean",
    )
    parser.add_argument(
        "--normalization",
        choices=["standard", "minmax", "maxabs", "robust", "none"],
        default="standard",
    )
    parser.add_argument(
        "--prototype-strategy",
        choices=["mean", "median", "nearest"],
        default="mean",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runtime" / "reports")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_path, pdf_path = create_report_example(
        feature_index=args.feature_index,
        metric=args.metric,
        normalization=args.normalization,
        prototype_strategy=args.prototype_strategy,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )
    log_saved_report(html_path, pdf_path)


def create_report_example(
    *,
    feature_index: int = 0,
    metric: str = "euclidean",
    normalization: str = "standard",
    prototype_strategy: str = "mean",
    output_dir: Path | str = PROJECT_ROOT / "runtime" / "reports",
    random_state: int = 42,
    dataset_source: str = DATASET_SOURCE,
) -> tuple[Path, Path]:
    dataframe = build_dataset(random_state=random_state)
    feature_columns = [column for column in dataframe.columns if column != TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe[feature_columns],
        dataframe[TARGET_NAME],
        test_size=0.35,
        stratify=dataframe[TARGET_NAME],
        random_state=random_state,
    )
    model = EtalonClassifier(
        metric=metric,
        normalization=normalization,
        prototype_strategy=prototype_strategy,
    ).fit(X_train, y_train)
    return (
        report_for(model)
        .with_training_data(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_columns,
            target_name=TARGET_NAME,
        )
        .with_test_data(
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_columns,
            target_name=TARGET_NAME,
        )
        .with_options(
            dataset_source=dataset_source,
            feature_index=feature_index,
            random_state=random_state,
        )
        .save(output_dir, stem="etalon_classification_report")
    )


def build_dataset(*, random_state: int) -> pd.DataFrame:
    return build_classification_dataset(random_state=random_state)


if __name__ == "__main__":
    main()
