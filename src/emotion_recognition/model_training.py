import pandas as pd
from sklearn.model_selection import RandomizedSearchCV


def input_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    pass


def get_best_model(X: pd.DataFrame, y: pd.DataFrame, parameters: dict, cv: int = 5) -> object:
    models_scores = {}

    for model_name in parameters.keys():
        single_model = parameters[model_name]["model"]
        single_model_parameters = parameters[model_name]["parameters"]

        best_single_model, best_single_model_score = _get_single_best_model(
            X, y, single_model, single_model_parameters, cv
        )
        models_scores[best_single_model] = best_single_model_score

    return max(models_scores, key=models_scores.get)


def _get_single_best_model(
    X: pd.DataFrame, y: pd.DataFrame, single_model: object, single_model_parameters: dict, cv: int = 5
) -> tuple[object, float]:
    clf = RandomizedSearchCV(single_model, single_model_parameters, cv=cv)
    search = clf.fit(X, y)

    return search.best_estimator_, search.best_score_
