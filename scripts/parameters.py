from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

confidence_threshold = 0.7

training_parameters = {
    "random_forest": {
        "model": RandomForestClassifier,
        "parameters": {
            "n_estimators": range(10, 100, 5),
            "max_depth": range(3, 10, 1),
        },
    },
    "svc": {
        "model": SVC,
        "parameters": {
            "kernel": ["linear", "poly", "rbf"],
            "C": loguniform(1e-3, 1),
        },
    },
}
