import pandas as pd

from emotion_recognition.data_transformers.base_transformer import BaseTransformerFixPP
from emotion_recognition.utils import calculate_basic_stats, detect_saccades, merge_columns


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

        saccades_left = detect_saccades(df[df["eye_id"] == 0], "left")
        saccades_right = detect_saccades(df[df["eye_id"] == 1], "right")
        saccades = pd.concat([saccades_left, saccades_right])

        df = df.merge(saccades, how="left", left_on=["timestamp", "eye_id"], right_on=["start_time", "eye_id"])

        features_diameter = calculate_basic_stats(df, "diameter")
        features_saccade_duration = calculate_basic_stats(df, "saccade_duration")
        features_saccade_amplitude_cartesian = calculate_basic_stats(df, "saccade_amplitude_cartesian")

        return features_diameter.merge(features_saccade_duration, how="outer", on=merge_columns).merge(
            features_saccade_amplitude_cartesian, how="outer", on=merge_columns
        )
