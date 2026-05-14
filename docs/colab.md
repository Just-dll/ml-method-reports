# Google Colab

Open in Colab badge placeholder:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Just-dll/ml-method-reports/blob/main/notebooks/00_colab_quickstart.ipynb)

Install cell after PyPI publication:

```python
!pip install "ml-method-reports[notebook]"
```

Before PyPI publication:

```python
# !pip install "git+https://github.com/Just-dll/ml-method-reports.git"
```

Minimal usage:

```python
from ml_method_reports import report_for

report_for(model).with_data(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names,
).display()
```

Use `.save("runtime/reports/example")` to export HTML/PDF files in the Colab runtime.
