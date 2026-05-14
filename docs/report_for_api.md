# report_for API

The fluent API starts from a fitted model:

```python
request = report_for(model)
```

Common methods:

```python
report_for(model)
    .with_training_data(...)
    .with_test_data(...)
    .with_options(...)
    .as_generic()
    .with_builder(...)
    .build()
    .display()
    .save_html("runtime/reports/example.html")
    .save_pdf("runtime/reports/example.pdf")
    .save("runtime/reports/example")
```

`with_training_data(...)` and `with_test_data(...)` are the preferred methods for public examples because they make the report context explicit. `with_data(...)` is also available as a shortcut when you already have train and test arrays together.

`.build()` returns `ExperimentReport`. `.display()` renders inline in notebooks. `.save_html(...)` writes only HTML. `.save_pdf(...)` writes only PDF. `.save(...)` keeps the existing bundle behavior and writes both HTML and PDF files.

Use `.with_options(...)` for report-specific controls such as selected sample indices, scaling metadata, dataset source, or progress output.
