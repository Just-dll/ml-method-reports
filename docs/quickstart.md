# Quickstart

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from ml_method_reports import report_for

X, y = make_classification(n_samples=200, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

report_for(model).with_data(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=["f1", "f2", "f3", "f4"],
).save("runtime/reports/knn_example")
```
