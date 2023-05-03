import re
from pathlib import Path

import numpy as np
import pandas as pd
from constants import DATA_INTERMEDIATE_PATHS, DATA_RAW_PATHS, TRANSFORMERS
from tqdm import tqdm

from emotion_recognition.data_transformers.base_transformer import BaseTransformer


def prepare_key_pictures():
    key_picture = (
        pd.read_excel(DATA_RAW_PATHS["key_pictures"])
        .replace("n/d", None)
        .astype({"valence_norm": np.float64, "arousal_norm": np.float64})
        .loc[:, ["name", "trigger", "valence_norm", "arousal_norm"]]
        .rename(columns={"name": "image_name", "trigger": "image_id"})
    )
    key_picture_not_hi = key_picture[~key_picture["image_name"].str.startswith("hi_")].drop(columns="image_id")
    key_picture_hi = key_picture_not_hi.assign(image_name="hi_" + key_picture_not_hi["image_name"])
    key_picture_filled = pd.concat([key_picture_not_hi, key_picture_hi])
    key_picture.loc[:, ["image_name", "image_id"]].merge(key_picture_filled, how="left", on="image_name")

    key_picture.to_parquet(DATA_INTERMEDIATE_PATHS["key_pictures"])


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
    perform_base_cleaning()
    prepare_key_pictures()
