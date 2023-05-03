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

    @staticmethod
    def merge_with_key_pictures(df, df_key_pictures):
        return df.merge(df_key_pictures, how="left", on="image_name")
