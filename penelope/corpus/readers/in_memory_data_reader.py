from typing import Any, Dict, List, Tuple

import pandas as pd

from penelope.utility.file_utility import IndexOfSplitOrCallableOrRegExp, basenames, extract_filename_fields

from .interfaces import ICorpusReader


class InMemoryReader(ICorpusReader):

    """Text iterator that returns row-wise text documents from a Pandas DataFrame"""

    def __init__(self, data: List[Tuple[str, List[str]]], filename_fields: IndexOfSplitOrCallableOrRegExp):

        self.data = data
        self._filename_fields = filename_fields
        self._filenames = [x[0] for x in self.data]
        self._metadata = self._create_metadata(self._filenames)
        self._documents = pd.DataFrame(self._metadata)
        self.iterator = None

    def _create_metadata(self, filenames):
        return [
            {'filename': filename, **extract_filename_fields(filename, self._filename_fields)}
            for filename in basenames(filenames)
        ]

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
