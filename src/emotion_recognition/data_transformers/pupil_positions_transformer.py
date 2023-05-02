from emotion_recognition.data_transformers.base_transformer import BaseTransformerFixPP


class PupilPositionsTransformer(BaseTransformerFixPP):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = [
            "pupil_timestamp",
            "eye_id",
            "confidence",
            "norm_pos_x",
            "norm_pos_y",
            "diameter",
            "method",
            "model_confidence",
        ]
        self.rename_map = {"pupil_timestamp": "tm_start"}
        self.method = "2d c++"
