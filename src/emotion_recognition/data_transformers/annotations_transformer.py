from emotion_recognition.data_transformers.base_transformer import BaseTransformer


class AnnotationsTransformer(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = ["timestamp", "UnityTriggertrigger"]
        self.rename_map = {"UnityTriggertrigger": "image_id"}
