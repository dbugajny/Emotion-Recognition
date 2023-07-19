import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger()


def input_missing_values(X: pd.DataFrame, strategy: str) -> pd.DataFrame:
    return SimpleImputer(missing_values=np.nan, strategy=strategy).fit_transform(X)


def get_best_model(
    X: pd.DataFrame, y: pd.DataFrame, model_hyperparameters: dict, cv: int = 5, n_iter: int = 10
) -> object:
    models_scores = {}

    for model_name in model_hyperparameters.keys():
        logging.info(f"Training {model_name} model")
        single_model = model_hyperparameters[model_name]["model"]
        single_model_hyperparameters = model_hyperparameters[model_name]["hyperparameters"]

        best_single_model, best_single_model_score = _get_single_best_model(
            X, y, single_model, single_model_hyperparameters, cv, n_iter
        )
        models_scores[best_single_model] = best_single_model_score

    return max(models_scores, key=models_scores.get)


def _get_single_best_model(
    X: pd.DataFrame, y: pd.DataFrame, single_model: object, single_model_hyperparameters: dict, cv: int, n_iter: int
) -> tuple[object, float]:
    clf = RandomizedSearchCV(single_model, single_model_hyperparameters, cv=cv, n_iter=n_iter)
    search = clf.fit(X, y)

    return search.best_estimator_, search.best_score_
