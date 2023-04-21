import pandas as pd

from emotion_recognition.data_transformers.base_transformer import BaseTransformer
from constants import TRANSFORMERS, DATA_RAW_PATHS, DATA_INTERMEDIATE_PATHS


def perform_base_cleaning(df: pd.DataFrame, transformer: BaseTransformer, filename: str) -> pd.DataFrame:
    df = transformer.perform_base_cleaning(df)
    df = transformer.extract_person_id(df, filename)

    return df


def main() -> None:
    for transformer, raw_path, intermediate_path in zip(TRANSFORMERS, DATA_RAW_PATHS, DATA_INTERMEDIATE_PATHS):
        intermediate_path.mkdir(parents=True, exist_ok=True)
        for filepath in raw_path.iterdir():
            df = pd.read_csv(filepath)
            df = perform_base_cleaning(df, transformer, filepath.stem)
            df.to_parquet(intermediate_path)


if __name__ == "__main__":
    main()
