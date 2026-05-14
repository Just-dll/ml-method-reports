from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_method_reports.reporting import report_for  # noqa: E402
from report_example_utils import (  # noqa: E402
    DATASET_SOURCE,
    FEATURE_NAMES,
    TARGET_NAME,
    build_classification_dataset,
    log_saved_report,
    split_classification_dataset,
    standard_scale_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an SVC educational report.")
    parser.add_argument("--kernel", choices=["rbf", "linear"], default="rbf")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runtime" / "reports" / "svc_example"
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    html_path, pdf_path = create_report_example(
        output_dir=args.output_dir,
        random_state=args.random_state,
        kernel=args.kernel,
    )
    log_saved_report(html_path, pdf_path)


def create_report_example(
    *,
    output_dir: Path | str,
    random_state: int = 42,
    kernel: str = "rbf",
) -> tuple[Path, Path]:
    dataframe = build_classification_dataset(random_state=random_state)
    X_train, X_test, y_train, y_test = split_classification_dataset(
        dataframe, random_state=random_state
    )
    X_train_scaled, X_test_scaled, scaling_params = standard_scale_frames(X_train, X_test)
    model = SVC(kernel=kernel, probability=True, random_state=random_state).fit(
        X_train_scaled, y_train
    )
    return (
        report_for(model)
        .with_training_data(
            X_train=X_train_scaled,
            y_train=y_train,
            feature_names=FEATURE_NAMES,
            target_name=TARGET_NAME,
        )
        .with_test_data(
            X_test=X_test_scaled,
            y_test=y_test,
            feature_names=FEATURE_NAMES,
            target_name=TARGET_NAME,
        )
        .with_options(
            dataset_source=DATASET_SOURCE,
            scaling_method="standard",
            scaling_params=scaling_params,
        )
        .save(output_dir, stem="svc_report")
    )


if __name__ == "__main__":
    main()
