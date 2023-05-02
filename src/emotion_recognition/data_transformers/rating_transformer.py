from emotion_recognition.data_transformers.base_transformer import BaseTransformer


class RatingTransformer(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = ["image9", "Valence_rating.response", "Arousal_rating.response"]
        self.rename_map = {}

    def merge_with_key_pictures(self, df, df_key_pictures):
        return df.merge(df_key_pictures, how="left", left_on="image9", right_on="name").loc[
            :, self.columns_to_keep + ["trigger", "valence_norm", "arousal_norm"]
        ]
