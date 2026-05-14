# Progressive Reports

Reports become richer as more context is provided. This lets you start with a fitted model and add data only when it is available.

| Input provided | What the toolkit can show |
|---|---|
| model only | model type, parameters, available artifacts |
| model + training data | training summary, class distribution, training artifacts |
| model + X_test | predictions, scores/probabilities, selected sample explanation |
| model + X_test + y_test | metrics, confusion matrix, error analysis |
| full train/test data | full educational report with visualizations |

Use `with_training_data(...)`, `with_test_data(...)`, or `with_data(...)` to provide context.
