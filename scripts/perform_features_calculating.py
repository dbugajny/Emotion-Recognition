from pathlib import Path

import pandas as pd
from constants import DATA_FEATURE_PATHS, DATA_PRIMARY_PATHS, TRANSFORMERS
from hyperparameters import CONFIDENCE_THRESHOLD
from tqdm import tqdm


def perform_features_calculating() -> None:
    data_sources = ["fixations", "pupil_positions", "bitalino"]

    for data_source in data_sources:
        DATA_FEATURE_PATHS[data_source].mkdir(parents=True, exist_ok=True)

        for filepath in tqdm(
            DATA_PRIMARY_PATHS[data_source].iterdir(),
            total=len(list(DATA_PRIMARY_PATHS[data_source].iterdir())),
            desc=f"Processing {DATA_PRIMARY_PATHS[data_source].stem} directory.",
        ):
            df = pd.read_parquet(filepath)
            if df.empty:
                continue

            if data_source == "bitalino":
                df = TRANSFORMERS[data_source].calculate_features(df)
            else:
                df = TRANSFORMERS[data_source].calculate_features(df, CONFIDENCE_THRESHOLD)

            df.to_parquet(DATA_FEATURE_PATHS[data_source] / Path(filepath.name))


if __name__ == "__main__":
    perform_features_calculating()
