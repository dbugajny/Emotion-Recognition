import pandas as pd
from functools import reduce
from constants import DATA_FEATURE_PATHS, DATA_MODEL_INPUT_PATHS
from tqdm import tqdm
from emotion_recognition.data_transformers.utils import merge_columns


def perform_features_merging() -> None:
    data_sources = [
        "fixations",
        "pupil_positions",
        "bitalino",
    ]

    lst_features_merged = []

    for data_source in data_sources:
        features_concatenated = pd.DataFrame()

        for filepath in tqdm(
            DATA_FEATURE_PATHS[data_source].iterdir(),
            total=len(list(DATA_FEATURE_PATHS[data_source].iterdir())),
            desc=f"Processing {DATA_FEATURE_PATHS[data_source].stem} directory.",
        ):
            df = pd.read_parquet(filepath)
            features_concatenated = pd.concat([features_concatenated, df])

        lst_features_merged.append(features_concatenated)

    final_merge = reduce(lambda left, right: pd.merge(left, right, on=merge_columns, how="outer"), lst_features_merged)
    final_merge.to_parquet(DATA_MODEL_INPUT_PATHS["features_merged"])


if __name__ == "__main__":
    perform_features_merging()
