import re
from pathlib import Path

import pandas as pd
from constants import DATA_INTERMEDIATE_PATHS, DATA_PRIMARY_PATHS, TRANSFORMERS
from tqdm import tqdm


def merge_ratings_with_key_pictures():
    key_pictures = pd.read_parquet(DATA_INTERMEDIATE_PATHS["key_pictures"])
    rating_transformer = TRANSFORMERS["rating"]

    DATA_PRIMARY_PATHS["rating"].mkdir(parents=True, exist_ok=True)

    for filepath in tqdm(
        DATA_INTERMEDIATE_PATHS["rating"].iterdir(),
        total=len(list(DATA_INTERMEDIATE_PATHS["rating"].iterdir())),
        desc=f"Processing {DATA_INTERMEDIATE_PATHS['rating'].stem} directory.",
    ):
        df_rating = pd.read_parquet(filepath)
        df_rating = rating_transformer.merge_with_key_pictures(df_rating, key_pictures)
        df_rating.to_parquet(DATA_PRIMARY_PATHS["rating"] / Path(filepath.name))


def merge_fix_pp_with_annotations_and_ratings():
    data_sources = ["fixations", "pupil_positions"]
    rating_filepaths = list(DATA_PRIMARY_PATHS["rating"].iterdir())

    for data_source in data_sources:
        DATA_PRIMARY_PATHS[data_source].mkdir(parents=True, exist_ok=True)

        for filepath in tqdm(
            DATA_INTERMEDIATE_PATHS[data_source].iterdir(),
            total=len(list(DATA_INTERMEDIATE_PATHS[data_source].iterdir())),
            desc=f"Processing {DATA_INTERMEDIATE_PATHS[data_source].stem} directory.",
        ):
            df_annotations_filepath = DATA_INTERMEDIATE_PATHS["annotations"] / re.sub(r"_(f|p)\.", "_a.", filepath.name)
            df_annotations = pd.read_parquet(df_annotations_filepath)

            person_id = re.search(r"[Ss]\d{2,3}", filepath.stem).group(0).upper()

            df_ratings_filepath = [r_fp for r_fp in rating_filepaths if person_id in r_fp.stem.upper()]
            df_ratings = pd.read_parquet(df_ratings_filepath[0])

            df = pd.read_parquet(filepath)
            df = TRANSFORMERS[data_source].merge_with_annotations_and_ratings(df, df_annotations, df_ratings)
            df.to_parquet(DATA_PRIMARY_PATHS[data_source] / Path(filepath.name))


def merge_bitalino_with_ratings():
    rating_filepaths = list(DATA_PRIMARY_PATHS["rating"].iterdir())

    DATA_PRIMARY_PATHS["bitalino"].mkdir(parents=True, exist_ok=True)

    for filepath in tqdm(
        DATA_INTERMEDIATE_PATHS["bitalino"].iterdir(),
        total=len(list(DATA_INTERMEDIATE_PATHS["bitalino"].iterdir())),
        desc=f"Processing {DATA_INTERMEDIATE_PATHS['bitalino'].stem} directory.",
    ):
        person_id = re.search(r"[Ss]\d{2,3}", filepath.stem).group(0).upper()

        df_ratings_filepath = [r_fp for r_fp in rating_filepaths if person_id in r_fp.stem.upper()]
        df_ratings = pd.read_parquet(df_ratings_filepath[0])

        df = pd.read_parquet(filepath)
        df = TRANSFORMERS["bitalino"].merge_with_ratings(df, df_ratings)
        df.to_parquet(DATA_PRIMARY_PATHS["bitalino"] / Path(filepath.name))


if __name__ == "__main__":
    merge_ratings_with_key_pictures()
    merge_fix_pp_with_annotations_and_ratings()
    merge_bitalino_with_ratings()
