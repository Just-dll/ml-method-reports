# Installation

There is more than one way to use `ml-method-reports`: install the package from PyPI, install it directly from GitHub, or run it from a local editable checkout during development.

Install the package:

```bash
pip install ml-method-reports
```

For notebooks:

```bash
pip install "ml-method-reports[notebook]"
```

Install directly from GitHub:

```bash
pip install "git+https://github.com/Just-dll/ml-method-reports.git"
```

For development:

```bash
git clone https://github.com/Just-dll/ml-method-reports.git
cd ml-method-reports
pip install -e ".[dev,notebook]"
```

Editable install combinations:

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[notebook]"
pip install -e ".[dev,notebook]"
```

All installation methods expose the same Python import package:

```python
from ml_method_reports import EtalonClassifier, report_for
from ml_method_reports.reporting import ExperimentReport, ReportSection
```
