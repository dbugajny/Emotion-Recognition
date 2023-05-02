from emotion_recognition.data_transformers.base_transformer import BaseTransformer


class RatingTransformer(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = ["image9", "Valence_rating.response", "Arousal_rating.response"]
        self.rename_map = {}
