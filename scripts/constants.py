from pathlib import Path

from emotion_recognition.data_transformers.annotations_transformer import AnnotationsTransformer
from emotion_recognition.data_transformers.bitalino_transformer import BitalinoTransformer
from emotion_recognition.data_transformers.fixations_transformer import FixationsTransformer
from emotion_recognition.data_transformers.pupil_positions_transformer import PupilPositionsTransformer
from emotion_recognition.data_transformers.rating_transformer import RatingTransformer

TRANSFORMERS = {
    "annotations": AnnotationsTransformer(),
    "bitalino": BitalinoTransformer(),
    "fixations": FixationsTransformer(),
    "pupil_positions": PupilPositionsTransformer(),
    "rating": RatingTransformer(),
}

KEY_PICTURE_PATH = Path("data/01_raw/__documents/key_pic.xlsx")

DATA_RAW_PATHS = {
    "annotations": Path("data/01_raw/annotations"),
    "bitalino": Path("data/01_raw/bitalino"),
    "fixations": Path("data/01_raw/fixations"),
    "pupil_positions": Path("data/01_raw/pupil_positions"),
    "rating": Path("data/01_raw/rating"),
    "key_pictures": Path("data/01_raw/__documents/key_pic.xlsx"),
}

DATA_INTERMEDIATE_PATHS = {
    "annotations": Path("data/02_intermediate/annotations"),
    "bitalino": Path("data/02_intermediate/bitalino"),
    "fixations": Path("data/02_intermediate/fixations"),
    "pupil_positions": Path("data/02_intermediate/pupil_positions"),
    "rating": Path("data/02_intermediate/rating"),
    "key_pictures": Path("data/02_intermediate/key_pic.parquet"),
}

DATA_PRIMARY_PATHS = {
    "annotations": Path("data/03_primary/annotations"),
    "bitalino": Path("data/03_primary/bitalino"),
    "fixations": Path("data/03_primary/fixations"),
    "pupil_positions": Path("data/03_primary/pupil_positions"),
    "rating": Path("data/03_primary/rating"),
}

DATA_FEATURE_PATHS = {
    "annotations": Path("data/04_feature/annotations"),
    "bitalino": Path("data/04_feature/bitalino"),
    "fixations": Path("data/04_feature/fixations"),
    "pupil_positions": Path("data/04_feature/pupil_positions"),
    "rating": Path("data/04_feature/rating"),
}

DATA_MODEL_INPUT_PATHS = {
    "features_merged": Path("data/05_model_input/features_merged.parquet"),
}
