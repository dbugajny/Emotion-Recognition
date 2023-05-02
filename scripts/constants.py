from pathlib import Path

from emotion_recognition.data_transformers.annotations_transformer import AnnotationsTransformer
from emotion_recognition.data_transformers.bilatino_transformer import BilatinoTransformer
from emotion_recognition.data_transformers.fixations_transformer import FixationsTransformer
from emotion_recognition.data_transformers.pupil_positions_transformer import PupilPositionsTransformer
from emotion_recognition.data_transformers.rating_transformer import RatingTransformer

DATA_SOURCES = [
    "annotations",
    "bitalino",
    "fixations",
    "pupil_positions",
    "rating",
]

TRANSFORMERS = {
    "annotations": AnnotationsTransformer(),
    "bitalino": BilatinoTransformer(),
    "fixations": FixationsTransformer(),
    "pupil_positions": PupilPositionsTransformer(),
    "rating": RatingTransformer(),
}

DATA_RAW_PATHS = {
    "annotations": Path("data/01_raw/annotations"),
    "bitalino": Path("data/01_raw/bitalino"),
    "fixations": Path("data/01_raw/fixations"),
    "pupil_positions": Path("data/01_raw/pupil_positions"),
    "rating": Path("data/01_raw/rating"),
}

DATA_INTERMEDIATE_PATHS = {
    "annotations": Path("data/02_intermediate/annotations"),
    "bitalino": Path("data/02_intermediate/bitalino"),
    "fixations": Path("data/02_intermediate/fixations"),
    "pupil_positions": Path("data/02_intermediate/pupil_positions"),
    "rating": Path("data/02_intermediate/rating"),
}

DATA_PRIMARY_PATHS = {
    "annotations": Path("data/03_primary/annotations"),
    "bitalino": Path("data/03_primary/bitalino"),
    "fixations": Path("data/03_primary/fixations"),
    "pupil_positions": Path("data/03_primary/pupil_positions"),
    "rating": Path("data/03_primary/rating"),
}

KEY_PICTURE_PATH = Path("data/01_raw/__documents/key_pic.xlsx")
