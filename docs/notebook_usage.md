# Notebook Usage

Install the notebook extra:

```bash
pip install "ml-method-reports[notebook]"
```

Render a report inline:

```python
report_for(model) \
    .with_training_data(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
    ) \
    .with_test_data(
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    ) \
    .display()
```

`.with_training_data(...)` and `.with_test_data(...)` are the preferred way to make report context explicit. `.with_data(...)` is available as a compact shortcut when you already have all arrays together.

`.display()` renders inline in Jupyter Notebook or Google Colab. `.build()` returns an `ExperimentReport` object for custom rendering, inspection, or tests.

Export both HTML and PDF:

```python
request.save("runtime/reports/example")
```

Export only one format:

```python
request.save_html("runtime/reports/example.html")
request.save_pdf("runtime/reports/example.pdf")
```

In Colab, install the package in the first cell before imports.
