# Google Colab

Open in Colab badge placeholder:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Just-dll/ml-method-reports/blob/main/notebooks/00_colab_quickstart.ipynb)

Install the library into the active Colab runtime before importing it:

```python
%pip install -q "ml-method-reports[notebook] @ git+https://github.com/Just-dll/ml-method-reports.git"
```

After a PyPI release is available, the shorter PyPI install form is:

```python
%pip install -q "ml-method-reports[notebook]"
```

Both methods expose the same import package:

```python
from ml_method_reports import EtalonClassifier, report_for
from ml_method_reports.reporting import ExperimentReport, ReportSection
```

Minimal usage:

```python
from ml_method_reports import report_for

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

Use `.save("runtime/reports/example")` to export the HTML/PDF bundle in the Colab runtime. Use `.save_html("runtime/reports/example.html")` or `.save_pdf("runtime/reports/example.pdf")` when you only need one format.
