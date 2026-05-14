# Notebook Usage

Install the notebook extra:

```bash
pip install "ml-method-reports[notebook]"
```

Render a report inline:

```python
report_for(model).with_data(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names,
).display()
```

`.display()` renders inline in Jupyter Notebook or Google Colab. `.save(...)` exports HTML/PDF. `.build()` returns an `ExperimentReport` object for custom rendering, inspection, or tests.

In Colab, install the package in the first cell before imports.
