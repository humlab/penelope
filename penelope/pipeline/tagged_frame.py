
from typing import Dict
import pandas as pd

TaggedFrame = pd.core.api.DataFrame

class TaggedDocumentFrame():
    """A tagged document represented as a pandas dataframe"""
    def __init__(self, tagged_frame: pd.DataFrame, id2token: Dict[str, int]=None, is_numeric: bool=False):

        self.tagged_frame = tagged_frame
        self.id2token = id2token
        self.is_numeric = is_numeric

        if self.is_numeric and not self.id2token:
            raise ValueError("Numeric TaggedDocumentFrame must have a vocabulary (id2token mapping)")

