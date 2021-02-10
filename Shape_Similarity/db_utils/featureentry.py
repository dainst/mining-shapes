from typing import List, NamedTuple


class FeatureEntry(NamedTuple):
    id: str
    feature_vec: List[float]
