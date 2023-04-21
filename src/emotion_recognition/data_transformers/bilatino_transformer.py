from emotion_recognition.data_transformers.base_transformer import BaseTransformer


class BilatinoTransformer(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = []
        self.rename_map = {}
