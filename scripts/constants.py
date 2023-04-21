from pathlib import Path

from emotion_recognition.data_transformers.annotations_transformer import AnnotationsTransformer
from emotion_recognition.data_transformers.bilatino_transformer import BilatinoTransformer
from emotion_recognition.data_transformers.fixations_transformer import FixationsTransformer
from emotion_recognition.data_transformers.pupil_positions_transformer import PupilPositionsTransformer
from emotion_recognition.data_transformers.rating_transformer import RatingTransformer

TRANSFORMERS = [
    AnnotationsTransformer(),
    BilatinoTransformer(),
    FixationsTransformer(),
    PupilPositionsTransformer(),
    RatingTransformer(),
]

DATA_RAW_PATHS = [
    Path("../data/01_raw/annotations"),
    Path("../data/01_raw/bitalino"),
    Path("../data/01_raw/fixations"),
    Path("../data/01_raw/pupil_positions"),
    Path("../data/01_raw/rating"),
]

DATA_INTERMEDIATE_PATHS = [
    Path("../data/02_intermediate/annotations"),
    Path("../data/02_intermediate/bitalino"),
    Path("../data/02_intermediate/fixations"),
    Path("../data/02_intermediate/pupil_positions"),
    Path("../data/02_intermediate/rating"),
]
