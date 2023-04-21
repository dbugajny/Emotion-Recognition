from emotion_recognition.data_transformers.base_transformer import BaseTransformerFixAnn


class FixationsTransformer(BaseTransformerFixAnn):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = ["start_timestamp", "norm_pos_x", "norm_pos_y", "dispersion", "confidence", "method"]
        self.rename_map = {}
        self.method = "2d c++"
