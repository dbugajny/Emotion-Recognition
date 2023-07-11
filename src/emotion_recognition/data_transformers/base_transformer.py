import re

import pandas as pd


class BaseTransformer:
    def __init__(self) -> None:
        self.columns_to_keep = []
        self.rename_map = {}

    def perform_base_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, self.columns_to_keep].rename(columns=self.rename_map).drop_duplicates()

    @staticmethod
    def extract_person_id(df: pd.DataFrame, filename: str) -> pd.DataFrame:
        return df.assign(person_id=re.search(r"[Ss]\d{2,3}", filename).group(0).upper())


class BaseTransformerFixPP(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.method = []

    def perform_base_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().perform_base_cleaning(df)
        return df[df["method"] == self.method]

    @staticmethod
    def merge_with_annotations_and_ratings(df, df_annotations, df_ratings):
        return pd.merge_asof(df, df_annotations.drop(columns="person_id"), on="timestamp").merge(
            df_ratings, how="left", on=["image_id", "person_id"]
        )

    @staticmethod
    def calculate_basic_stats(df: pd.DataFrame, feature_column: str) -> pd.DataFrame:
        columns = ["person_id", "image_id", "valence_rating", "arousal_rating", "valence_norm", "arousal_norm"]
        kurtosis = (
            df.groupby(columns,  group_keys=True)[feature_column]
            .apply(pd.DataFrame.kurt)
            .reset_index(name=f"kurtosis_{feature_column}")
        )

        return (
            df.groupby(columns,  group_keys=True)[feature_column]
            .agg(["mean", "median", "skew"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"mean_{feature_column}",
                    "median": f"median_{feature_column}",
                    "skew": f"skew_{feature_column}",
                }
            )
            .merge(kurtosis, how="outer", on=columns)
        )
