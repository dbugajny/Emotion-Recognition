import pandas as pd

merge_columns = ["person_id", "image_id", "valence_rating", "arousal_rating", "valence_norm", "arousal_norm"]


def calculate_basic_stats(df: pd.DataFrame, feature_column: str) -> pd.DataFrame:
    kurtosis = (
        df.groupby(merge_columns, group_keys=True)[feature_column]
        .apply(pd.DataFrame.kurt)
        .reset_index(name=f"kurtosis_{feature_column}")
    )

    return (
        df.groupby(merge_columns, group_keys=True)[feature_column]
        .agg(["mean", "median", "skew"])
        .reset_index()
        .rename(
            columns={
                "mean": f"mean_{feature_column}",
                "median": f"median_{feature_column}",
                "skew": f"skew_{feature_column}",
            }
        )
        .merge(kurtosis, how="outer", on=merge_columns)
    )
