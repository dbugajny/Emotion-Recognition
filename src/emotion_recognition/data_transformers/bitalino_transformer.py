from emotion_recognition.data_transformers.base_transformer import BaseTransformer


class BitalinoTransformer(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = ["EKG", "light", "EDA", "HR", "trigger"]
        self.rename_map = {"trigger": "image_id"}

    @staticmethod
    def merge_with_ratings(df, df_ratings):
        return df.merge(df_ratings, how="left", on="image_id")
