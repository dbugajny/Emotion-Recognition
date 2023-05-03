import re

import pandas as pd


class BaseTransformer:
    def __init__(self) -> None:
        self.columns_to_keep = []
        self.rename_map = {}

    def perform_base_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, self.columns_to_keep].rename(columns=self.rename_map)

    @staticmethod
    def extract_person_id(df: pd.DataFrame, filename: str) -> pd.DataFrame:
        return df.assign(id_person=re.search(r"[Ss]\d{2,3}", filename).group(0).upper())


class BaseTransformerFixPP(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.method = []

    def perform_base_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().perform_base_cleaning(df)
        return df[df["method"] == self.method]

    @staticmethod
    def merge_with_annotations_and_ratings(df, df_annotations, df_ratings):
        return pd.merge_asof(df, df_annotations, left_on="tm_start", right_on="timestamp").merge(
            df_ratings, how="left", on="image_id"
        )
