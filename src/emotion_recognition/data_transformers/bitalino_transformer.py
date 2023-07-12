from emotion_recognition.data_transformers.base_transformer import BaseTransformer
from emotion_recognition.data_transformers.utils import calculate_basic_stats
import pandas as pd


class BitalinoTransformer(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = ["EKG", "light", "EDA", "HR", "trigger"]
        self.rename_map = {"trigger": "image_id"}
        self.cast_map = {"EKG": float, "light": float, "EDA": float, "HR": float, "trigger": int}

    @staticmethod
    def merge_with_ratings(df, df_ratings):
        return df.merge(df_ratings, how="left", on=["image_id", "person_id"])

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["person_id", "image_id"], how="any")
        features = calculate_basic_stats(df, "HR")

        return features
