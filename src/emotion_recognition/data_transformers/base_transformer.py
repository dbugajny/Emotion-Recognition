import re

import pandas as pd


class BaseTransformer:
    def __init__(self) -> None:
        self.columns_to_keep = []
        self.rename_map = {}

    def perform_base_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, self.columns_to_keep].rename(self.rename_map)

    @staticmethod
    def extract_person_id(df: pd.DataFrame, filename: str) -> pd.DataFrame:
        return df.assign(id_person=re.search(r"[Ss]\d{2,3}", filename))


class BaseTransformerFixAnn(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.method = []

    def perform_base_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().perform_base_cleaning(df)
        return df[df["method"] == self.method]
