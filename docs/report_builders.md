# Report Builders

Report builders convert a `ReportContext` into an `ExperimentReport`.

To add a new report builder:

1. Create a builder under `src/ml_method_reports/reporting/builders/`.
2. Accept a `ReportContext` in the constructor.
3. Return an `ExperimentReport` from `build()`.
4. Add an adapter when the builder should be auto-selected for a model type.
5. Add tests and an example script.

For one-off reports, pass a builder directly:

```python
report_for(model).with_builder(MyReportBuilder).build()
```

Use generic reports when no specialized adapter exists:

```python
report_for(model) \
    .as_generic() \
    .with_training_data(...) \
    .with_test_data(...) \
    .save("runtime/reports/generic")
```
