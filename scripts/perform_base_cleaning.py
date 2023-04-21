import re
from pathlib import Path

import pandas as pd
from constants import DATA_INTERMEDIATE_PATHS, DATA_RAW_PATHS, TRANSFORMERS

from emotion_recognition.data_transformers.base_transformer import BaseTransformer
import logging

logger = logging.getLogger(__name__)


def perform_base_cleaning(df: pd.DataFrame, transformer: BaseTransformer, filename: str) -> pd.DataFrame:
    df = transformer.perform_base_cleaning(df)
    df = transformer.extract_person_id(df, filename)

    return df


def main() -> None:
    for transformer, raw_path, intermediate_path in zip(TRANSFORMERS, DATA_RAW_PATHS, DATA_INTERMEDIATE_PATHS):
        logger.info(f"Processing {raw_path.stem} directory.")
        intermediate_path.mkdir(parents=True, exist_ok=True)
        for filepath in raw_path.iterdir():
            if filepath.suffix != ".csv" or not re.search(r"[Ss]\d{2,3}", filepath.stem):
                continue
            df = pd.read_csv(filepath)
            df = perform_base_cleaning(df, transformer, filepath.stem)
            df.to_parquet(intermediate_path / Path(filepath.stem).with_suffix(".parquet"))


if __name__ == "__main__":
    main()
