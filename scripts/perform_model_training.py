import pickle

import pandas as pd
from constants import DATA_MODEL_INPUT_PATHS, DATA_MODELS_PATHS
from hyperparameters import CV, MODEL_HYPERPARAMETERS, N_ITER, X_COLUMNS

from emotion_recognition.model_training import get_best_model, input_missing_values


def perform_model_training(y_column):
    dataset_train = pd.read_parquet(DATA_MODEL_INPUT_PATHS["dataset_train"])

    X_train = dataset_train.loc[:, X_COLUMNS]
    X_train = input_missing_values(X_train, "median")

    y_train = dataset_train.loc[:, y_column]

    best_model = get_best_model(X_train, y_train, MODEL_HYPERPARAMETERS, cv=CV, n_iter=N_ITER)

    with open(DATA_MODELS_PATHS[f"best_model_{y_column}"], "wb") as f:
        pickle.dump(best_model, f)


if __name__ == "__main__":
    perform_model_training("valence_rating")
    perform_model_training("arousal_rating")
