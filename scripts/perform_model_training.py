import pandas as pd
from constants import DATA_MODEL_INPUT_PATHS
from parameters import training_parameters

from emotion_recognition.model_training import get_best_model, input_missing_values


def perform_model_training():
    X_columns = [
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
    y_column = "valence_rating"

    dataset = pd.read_parquet(DATA_MODEL_INPUT_PATHS["features_merged"])
    X = dataset.loc[:, X_columns]
    X = input_missing_values(X)
    y = dataset.loc[:, [y_column]]

    best_model = get_best_model(X, y, training_parameters, cv=5)

    return best_model
