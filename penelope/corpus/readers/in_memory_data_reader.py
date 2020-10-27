from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from penelope.utility.file_utility import IndexOfSplitOrCallableOrRegExp, extract_filenames_fields

from .interfaces import ICorpusReader


class InMemoryReader(ICorpusReader):

    """Text iterator that returns row-wise text documents from a Pandas DataFrame"""

    def __init__(self, data: List[Tuple[str, List[str]]], filename_fields: Sequence[IndexOfSplitOrCallableOrRegExp]):

        self.data = data
        self._filename_fields = filename_fields
        self._filenames = [x[0] for x in self.data]
        self._metadata = self._create_metadata()
        self._documents = pd.DataFrame(self._metadata)
        self.iterator = None

    def _create_metadata(self) -> Sequence[Dict]:
        return extract_filenames_fields(filenames=self._filenames, filename_fields=self._filename_fields)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.data)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise

    @property
    def filenames(self) -> List[str]:
        return self._filenames

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        return self._metadata

    @property
    def documents(self) -> pd.DataFrame:
        return self._documents
