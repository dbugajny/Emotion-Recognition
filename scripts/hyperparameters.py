from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
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
        },
    },
    "svc": {
        "model": SVC(),
        "hyperparameters": {
            "kernel": ["linear", "poly", "rbf"],
            "C": loguniform(1e-3, 1),
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
