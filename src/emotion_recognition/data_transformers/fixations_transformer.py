from emotion_recognition.data_transformers.base_transformer import BaseTransformerFixPP
from emotion_recognition.data_transformers.utils import calculate_basic_stats
from emotion_recognition.data_transformers.utils import merge_columns
import pandas as pd
import numpy as np


class FixationsTransformer(BaseTransformerFixPP):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = ["start_timestamp", "norm_pos_x", "norm_pos_y", "dispersion", "confidence", "method"]
        self.rename_map = {"start_timestamp": "timestamp"}
        self.cast_map = {
            "start_timestamp": float,
            "norm_pos_x": float,
            "norm_pos_y": float,
            "dispersion": float,
            "confidence": float,
            "method": str,
        }
        self.method = "3d gaze"

    @staticmethod
    def calculate_features(df: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
        df = df.dropna(subset=["person_id", "image_id"], how="any")
        df_features = df[df["confidence"] >= confidence_threshold].copy()
        df_features["fixation_duration"] = np.roll(df_features["timestamp"].diff(), -1)
        features = calculate_basic_stats(df_features, "fixation_duration")
        fixations_count = df_features.groupby(merge_columns)["timestamp"].count().reset_index(name="n_fixations")
        features = features.merge(fixations_count, how="outer", on=merge_columns)

        return features
