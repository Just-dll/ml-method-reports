# EtalonClassifier

`EtalonClassifier` is the custom model included in this project. It is intended for educational method-of-etalons experiments.

It supports distance-based prediction using class prototypes. Available options include metrics such as Euclidean, Manhattan, Chebyshev, and cosine distance; normalization strategies such as standard, min-max, max-abs, robust, and none; and prototype strategies such as mean, median, and nearest representative.

The report can show class etalons, distances from samples to etalons, selected predictions, and pseudo-probabilities derived from distances.

Unlike KNN, which votes from nearby training samples, the etalon method compares samples with class-level prototypes.
