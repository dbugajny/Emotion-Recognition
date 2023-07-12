from emotion_recognition.data_transformers.base_transformer import BaseTransformerFixPP
from emotion_recognition.data_transformers.utils import calculate_basic_stats
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
        self.cast_map = {
            "timestamp": float,
            "eye_id": float,
            "confidence": float,
            "norm_pos_x": float,
            "norm_pos_y": float,
            "diameter": float,
            "method": str,
        }
        self.method = "2d c++"

    @staticmethod
    def calculate_features(df: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
        df = df.dropna(subset=["person_id", "image_id"], how="any")
        df = df[df["confidence"] >= confidence_threshold].copy()

        features = calculate_basic_stats(df, "diameter")

        return features
