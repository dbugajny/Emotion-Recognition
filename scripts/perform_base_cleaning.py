import re
from pathlib import Path

import numpy as np
import pandas as pd
from constants import DATA_INTERMEDIATE_PATHS, DATA_RAW_PATHS, TRANSFORMERS
from tqdm import tqdm

from emotion_recognition.data_transformers.base_transformer import BaseTransformer


def prepare_key_pictures():
    key_pictures = (
        pd.read_excel(DATA_RAW_PATHS["key_pictures"])
        .replace("n/d", None)
        .astype({"valence_norm": np.float64, "arousal_norm": np.float64})
        .rename(columns={"name": "image_name", "trigger": "image_id"})
    )
    key_pictures_images = key_pictures.loc[:, ["image_name", "image_id"]]
    key_pictures_values = key_pictures.loc[:, ["valence_norm", "arousal_norm"]].dropna()
    key_pictures_values = pd.concat([key_pictures_values] * 2).reset_index(drop=True)
    key_pictures = pd.concat([key_pictures_images, key_pictures_values], axis=1)
    key_pictures.to_parquet(DATA_INTERMEDIATE_PATHS["key_pictures"])


def _single_base_cleaning(df: pd.DataFrame, transformer: BaseTransformer, filename: str) -> pd.DataFrame:
    df = transformer.perform_base_cleaning(df)
    df = transformer.extract_person_id(df, filename)

    return df


def perform_base_cleaning() -> None:
    data_sources = [
        "annotations",
        "bitalino",
        "fixations",
        "pupil_positions",
        "rating",
    ]

    for data_source in data_sources:
        DATA_INTERMEDIATE_PATHS[data_source].mkdir(parents=True, exist_ok=True)
        for filepath in tqdm(
            DATA_RAW_PATHS[data_source].iterdir(),
            total=len(list(DATA_RAW_PATHS[data_source].iterdir())),
            desc=f"Processing {DATA_RAW_PATHS[data_source].stem} directory.",
        ):
            if filepath.suffix != ".csv" or not re.search(r"[Ss]\d{2,3}", filepath.stem):
                continue
            df = pd.read_csv(filepath)
            df = _single_base_cleaning(df, TRANSFORMERS[data_source], filepath.stem)
            df.to_parquet(DATA_INTERMEDIATE_PATHS[data_source] / Path(filepath.stem).with_suffix(".parquet"))


if __name__ == "__main__":
    prepare_key_pictures()
    perform_base_cleaning()
