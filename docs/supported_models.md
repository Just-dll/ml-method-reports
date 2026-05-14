# Supported Models

| Report | Backend | What it explains |
|---|---|---|
| EtalonClassifier | custom implementation | class etalons, distances, selected prediction |
| KNN | scikit-learn | nearest neighbors, distances, voting |
| Logistic Regression | scikit-learn | coefficients, probabilities, feature influence |
| Decision Tree | scikit-learn | rules, decision path, feature importance |
| Random Forest | scikit-learn | ensemble summary, feature importance, errors |
| SVC | scikit-learn | support vectors, decision scores |
| KMeans | scikit-learn | cluster centers, distances, inertia |
| Agglomerative | scikit-learn | merge tree, cluster sizes, PCA projection |

Only `EtalonClassifier` is custom implemented. sklearn models are used as computational backends. The toolkit focuses on artifact extraction, explanation, visualization, notebook display, and report generation.
