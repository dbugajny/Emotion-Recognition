from scipy.stats import loguniform
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

CONFIDENCE_THRESHOLD = 0.7
CV = 5
N_ITER = 2

MODEL_HYPERPARAMETERS = {
    "random_forest": {
        "model": RandomForestClassifier(),
        "hyperparameters": {
            "n_estimators": range(10, 100, 5),
            "max_depth": range(3, 10, 1),
            "min_samples_split": range(2, 10, 1),
            "min_samples_leaf": range(1, 3, 1),
            "max_features": ["sqrt", "log2"],
        },
    },
    "svc": {
        "model": SVC(),
        "hyperparameters": {
            "kernel": ["linear", "poly", "rbf"],
            "C": loguniform(1e-3, 1),
        },
    },
    "adaboost": {
        "model": AdaBoostClassifier(),
        "hyperparameters": {
            "n_estimators": range(100, 501, 50),
        },
    },
}

X_COLUMNS = [
    "mean_fixation_duration",
    "median_fixation_duration",
    "skew_fixation_duration",
    "kurtosis_fixation_duration",
    "n_fixations",
    "mean_diameter",
    "median_diameter",
    "skew_diameter",
    "kurtosis_diameter",
    "mean_HR",
    "median_HR",
    "skew_HR",
    "kurtosis_HR",
]
