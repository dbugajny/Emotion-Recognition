from emotion_recognition.data_transformers.base_transformer import BaseTransformerFixPP
import pandas as pd
import numpy as np


class FixationsTransformer(BaseTransformerFixPP):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = ["start_timestamp", "norm_pos_x", "norm_pos_y", "dispersion", "confidence", "method"]
        self.rename_map = {"start_timestamp": "timestamp"}
        self.method = "3d gaze"

    def calculate_features(self, df: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
        df_features = df[df["confidence"] >= confidence_threshold].copy()
        df_features["fixation_duration"] = np.roll(df_features["timestamp"].diff(), -1)
        features = self.calculate_basic_stats(df_features, "fixation_duration")
        fixations_count = df_features.groupby(["image_id"])["timestamp"].count().reset_index(name="n_fixations")
        features = features.merge(fixations_count, how="outer", on="image_id")

        return features
