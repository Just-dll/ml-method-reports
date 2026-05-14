from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["signal_mean", "signal_slope", "texture_ratio", "stability_score"]
TARGET_NAME = "target"
DATASET_SOURCE = "sklearn.datasets.make_classification"


def build_classification_dataset(*, random_state: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=160,
        n_features=len(FEATURE_NAMES),
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        class_sep=1.35,
        flip_y=0.03,
        random_state=random_state,
    )
    dataframe = pd.DataFrame(X, columns=FEATURE_NAMES)
    dataframe[TARGET_NAME] = y
    return dataframe.round(6)


def split_classification_dataset(dataframe: pd.DataFrame, *, random_state: int):
    return train_test_split(
        dataframe[FEATURE_NAMES],
        dataframe[TARGET_NAME],
        test_size=0.35,
        stratify=dataframe[TARGET_NAME],
        random_state=random_state,
    )


def standard_scale_frames(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, {"mean": scaler.mean_, "scale": scaler.scale_}


def log_saved_report(html_path: str | Path, pdf_path: str | Path) -> None:
    print(
        f"[ml-method-reports] Report bundle ready: html={html_path}, pdf={pdf_path}",
        file=sys.stderr,
        flush=True,
    )
