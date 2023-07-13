import pandas as pd

from emotion_recognition.data_transformers.base_transformer import BaseTransformer


class RatingTransformer(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = ["image9", "Valence_rating.response", "Arousal_rating.response"]
        self.rename_map = {
            "image9": "image_name",
            "Valence_rating.response": "valence_rating",
            "Arousal_rating.response": "arousal_rating",
        }
        self.cast_map = {"image_name": str, "valence_rating": int, "arousal_rating": int}
        self.dropna_columns = ["image9", "Valence_rating.response", "Arousal_rating.response"]

    @staticmethod
    def merge_with_key_pictures(df, df_key_pictures):
        return (
            df.merge(df_key_pictures, how="left", on="image_name").drop(columns="image_name").dropna(subset="image_id")
        )

    def perform_base_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[(df[self.columns_to_keep] != "None").all(axis=1)].dropna(subset=self.dropna_columns, how="any")
        df = super().perform_base_cleaning(df)

        return df
