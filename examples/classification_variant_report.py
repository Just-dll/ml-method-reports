from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_EXAMPLE_PATH = PROJECT_ROOT / "examples" / "classification_report.py"

from report_example_utils import log_saved_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a classification variant report.")
    parser.add_argument("--variant", type=str, default="V")
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runtime" / "reports" / "classification_variant_V",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir == PROJECT_ROOT / "runtime" / "reports" / "classification_variant_V":
        output_dir = PROJECT_ROOT / "runtime" / "reports" / f"classification_variant_{args.variant}"

    html_path, pdf_path = create_report_example(
        variant=args.variant,
        feature_index=args.feature_index,
        metric=args.metric,
        normalization=args.normalization,
        prototype_strategy=args.prototype_strategy,
        output_dir=output_dir,
    )
    log_saved_report(html_path, pdf_path)


def create_report_example(
    *,
    variant: str = "V",
    feature_index: int = 0,
    metric: str = "euclidean",
    normalization: str = "standard",
    prototype_strategy: str = "mean",
    output_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    base_example = _load_base_example()
    base_example.FEATURE_NAMES = [
        "x1_variant_signal",
        "x2_variant_signal",
        "x3_variant_signal",
        "x4_variant_signal",
    ]
    base_example.build_dataset = lambda *, random_state: generate_classification_variant_dataset(
        variant=variant,
        random_state=random_state,
        feature_names=base_example.FEATURE_NAMES,
    )

    output = (
        Path(output_dir)
        if output_dir is not None
        else (PROJECT_ROOT / "runtime" / "reports" / f"classification_variant_{variant}")
    )
    return base_example.create_report_example(
        feature_index=feature_index,
        metric=metric,
        normalization=normalization,
        prototype_strategy=prototype_strategy,
        output_dir=output,
        random_state=_variant_random_state(variant),
        dataset_source=f"Classification variant {variant} synthetic generator",
    )


def generate_classification_variant_dataset(
    *,
    variant: str,
    random_state: int,
    feature_names: list[str],
) -> pd.DataFrame:
    # This synthetic dataset preserves the report workflow: normalization,
    # etalon centers, distance-based prediction, error estimates, one-feature
    # comparison, sklearn baselines, and report generation.
    X, y = make_classification(
        n_samples=160,
        n_features=len(feature_names),
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        class_sep=1.25,
        flip_y=0.04,
        random_state=random_state,
    )
    dataframe = pd.DataFrame(X, columns=feature_names)
    dataframe["target"] = y
    return dataframe.round(6)


def _load_base_example():
    spec = importlib.util.spec_from_file_location("etalon_classification_report", BASE_EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base etalon example from {BASE_EXAMPLE_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _variant_random_state(variant: str) -> int:
    return 100 + sum(ord(character) for character in variant)


if __name__ == "__main__":
    main()
