# report_for API

The fluent API starts from a fitted model:

```python
request = report_for(model)
```

Common methods:

```python
report_for(model)
    .with_data(...)
    .with_training_data(...)
    .with_test_data(...)
    .with_options(...)
    .as_generic()
    .with_builder(...)
    .build()
    .display()
    .save("runtime/reports/example")
```

`.build()` returns `ExperimentReport`. `.display()` renders inline in notebooks. `.save(...)` writes HTML and PDF files and returns their paths.

Use `.with_options(...)` for report-specific controls such as selected sample indices, scaling metadata, dataset source, or progress output.
