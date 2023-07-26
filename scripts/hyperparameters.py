from scipy.stats import loguniform
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR

CONFIDENCE_THRESHOLD = 0.7
CV = 5
N_ITER = 20

MODEL_HYPERPARAMETERS = {
    "random_forest": {
        "model": RandomForestRegressor(),
        "hyperparameters": {
            "n_estimators": range(10, 100, 5),
            "max_depth": range(3, 10, 1),
            "min_samples_split": range(2, 10, 1),
            "min_samples_leaf": range(1, 3, 1),
            "max_features": ["sqrt", "log2"],
        },
    },
    "svr": {
        "model": SVR(),
        "hyperparameters": {
            "kernel": ["linear", "poly", "rbf"],
            "C": loguniform(1e-3, 1),
        },
    },
    "adaboost": {
        "model": AdaBoostRegressor(),
        "hyperparameters": {
            "n_estimators": range(100, 1001, 10),
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
