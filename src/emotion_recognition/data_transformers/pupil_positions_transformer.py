from base_transformer import BaseTransformer


class PupilPositionsTransformer(BaseTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.columns_to_keep = []
        self.rename_map = {}


