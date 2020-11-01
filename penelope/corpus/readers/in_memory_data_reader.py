from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from penelope.utility import IndexOfSplitOrCallableOrRegExp, extract_filenames_fields, strip_paths

from .interfaces import ICorpusReader


class InMemoryReader(ICorpusReader):
    """Text iterator that returns row-wise text documents from an inline List of (doc-text tuples"""

    def __init__(self, data: List[Tuple[str, List[str]]], filename_fields: Sequence[IndexOfSplitOrCallableOrRegExp]):

        self.data = data

        self._filename_filter: List[str] = None
        self._filename_fields = filename_fields
        self._iterator = None

        self._all_filenames = [x[0] for x in self.data]
        self._all_metadata = self._create_all_metadata()
        self._documents: pd.DataFrame = pd.DataFrame(self._all_metadata)

    def _create_iterator(self):
        _filenames = self._get_filenames()
        return ((strip_paths(filename), document) for (filename, document) in self.data if filename in _filenames)

    def _create_all_metadata(self) -> Sequence[Dict[str, Any]]:
        return extract_filenames_fields(filenames=self._all_filenames, filename_fields=self._filename_fields)

    def _get_filenames(self):

        if self._filename_filter is None:
            return self._all_filenames

        return [filename for filename in self._all_filenames if filename in self._filename_filter]

    def _get_metadata(self, filenames):

        if self._filename_filter is None:
            return self._all_metadata

        return [metadata for metadata in self._all_metadata if metadata['filename'] in filenames]

    @property
    def filenames(self):
        return self._get_filenames()

    @property
    def metadata(self):
        return self._get_metadata(strip_paths(self._get_filenames()))

    @property
    def documents(self) -> pd.DataFrame:

        if self._filename_filter is None:
            return self._documents

        return self._documents[self._documents.filename.isin(self._filename_filter)]

    def apply_filter(self, filename_filter: List[str]):
        self._filename_filter = filename_filter

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = self._create_iterator()
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise
