# ml-method-reports

`ml-method-reports` is an educational Python toolkit for interactive exploration, explanation, visualization, and reporting of machine learning methods.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Just-dll/ml-method-reports/blob/main/notebooks/00_colab_quickstart.ipynb)

## What Problem It Solves

Students often run machine learning scripts and see only final metrics. This toolkit exposes the intermediate artifacts that explain how a model behaves: etalons, distances, nearest neighbors, voting, tree rules, coefficients, feature importance, support vectors, cluster centers, errors, and metrics.

## Features

- Custom `EtalonClassifier` for method-of-etalons experiments.
- sklearn-based educational explanations for common classifiers and clustering models.
- HTML and PDF export for reports that can be shared outside Python.
- Jupyter Notebook and Google Colab display with inline HTML rendering.
- Progressive report detail depending on the data you provide.
- Report builders for custom educational narratives.
- Extensible adapter/builder structure for future model reports.

## Installation

After publication to PyPI:

```bash
pip install ml-method-reports
```

For notebook usage after PyPI publication:

```bash
pip install "ml-method-reports[notebook]"
```

Before PyPI publication, install from GitHub:

```bash
pip install "git+https://github.com/Just-dll/ml-method-reports.git"
```

For development:

```bash
git clone https://github.com/Just-dll/ml-method-reports.git
cd ml-method-reports
pip install -e ".[dev,notebook]"
```

## Quick Start

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from ml_method_reports import report_for

X, y = make_classification(n_samples=200, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

report_for(model).with_data(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=["f1", "f2", "f3", "f4"],
).save("runtime/reports/knn_example")
```

## Notebook / Colab Usage

```python
report_for(model).with_data(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names,
).display()
```

`.display()` renders inline in a notebook, `.save(...)` exports HTML/PDF, and `.build()` returns an `ExperimentReport`.

## Progressive Report Detail

| Input provided | What the toolkit can show |
|---|---|
| model only | model type, parameters, available artifacts |
| model + training data | training summary, class distribution, training artifacts |
| model + X_test | predictions, scores/probabilities, selected sample explanation |
| model + X_test + y_test | metrics, confusion matrix, error analysis |
| full train/test data | full educational report with visualizations |

## Supported Reports

| Report | Backend | What it explains |
|---|---|---|
| EtalonClassifier | custom implementation | class etalons, distances, selected prediction |
| KNN | scikit-learn | nearest neighbors, distances, voting |
| Logistic Regression | scikit-learn | coefficients, probabilities, feature influence |
| Decision Tree | scikit-learn | rules, decision path, feature importance |
| Random Forest | scikit-learn | ensemble summary, feature importance, errors |
| SVC | scikit-learn | support vectors, decision scores |
| KMeans | scikit-learn | cluster centers, distances, inertia |
| Agglomerative | scikit-learn | merge tree, cluster sizes, PCA projection |

Only `EtalonClassifier` is custom implemented. sklearn models are used as computational backends; the project's value is artifact extraction, explanation, visualization, notebook display, and report generation.

## Examples and Notebooks

```text
examples/
notebooks/
docs/
```

Generate a local demo catalog:

```bash
python examples/generate_all_reports.py
```

## Documentation

See [`docs/`](docs/) for installation, quickstart, Colab, API, builder, supported model, and publishing notes. These Markdown pages are also suitable for GitHub Wiki pages.

## License

MIT. See [`LICENSE`](LICENSE).
