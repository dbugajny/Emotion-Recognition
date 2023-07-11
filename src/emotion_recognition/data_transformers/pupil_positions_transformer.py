from emotion_recognition.data_transformers.base_transformer import BaseTransformerFixPP
import pandas as pd


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
        ]
        self.rename_map = {"pupil_timestamp": "timestamp"}
        self.method = "2d c++"

    def calculate_features(self, df: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
        df = df[df["confidence"] >= confidence_threshold].copy()

        features = self.calculate_basic_stats(df, "diameter")

        return features
