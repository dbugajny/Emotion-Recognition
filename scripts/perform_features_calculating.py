from pathlib import Path
import pandas as pd

from constants import DATA_PRIMARY_PATHS, DATA_FEATURES_PATHS, TRANSFORMERS
from parameters import confidence_threshold
from tqdm import tqdm


def perform_features_calculating() -> None:
    data_sources = [
        "fixations",
        "pupil_positions",
    ]

    for data_source in data_sources:
        DATA_FEATURES_PATHS[data_source].mkdir(parents=True, exist_ok=True)

        for filepath in tqdm(
            DATA_PRIMARY_PATHS[data_source].iterdir(),
            total=len(list(DATA_PRIMARY_PATHS[data_source].iterdir())),
            desc=f"Processing {DATA_PRIMARY_PATHS[data_source].stem} directory.",
        ):
            df = pd.read_parquet(filepath)

            df = TRANSFORMERS[data_source].calculate_features(df, confidence_threshold)
            df.to_parquet(DATA_FEATURES_PATHS[data_source] / Path(filepath.name))


if __name__ == "__main__":
    perform_features_calculating()
