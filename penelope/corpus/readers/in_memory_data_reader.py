from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from penelope.corpus import metadata_to_document_index
from penelope.utility import extract_filenames_metadata, strip_paths

from .interfaces import ICorpusReader, TextReaderOpts


class InMemoryReader(ICorpusReader):
    """Text iterator that returns row-wise text documents from an inline List of (doc-text tuples"""

    def __init__(
        self,
        data: List[Tuple[str, List[str]]],
        reader_opts: TextReaderOpts,
    ):

        self.data = data

        self.reader_opts = reader_opts
        self._iterator = None

        self._all_filenames = [x[0] for x in self.data]
        self._all_metadata = self._create_all_metadata()
        self._document_index: pd.DataFrame = metadata_to_document_index(
            self._all_metadata, document_id_field=self.reader_opts.index_field
        )

    def _create_iterator(self):
        _filenames = self._get_filenames()
        return ((strip_paths(filename), document) for (filename, document) in self.data if filename in _filenames)

    def _create_all_metadata(self) -> Sequence[Dict[str, Any]]:
        return extract_filenames_metadata(
            filenames=self._all_filenames, filename_fields=self.reader_opts.filename_fields
        )

    def _get_filenames(self):

        if self.reader_opts.filename_filter is None:
            return self._all_filenames

        return [filename for filename in self._all_filenames if filename in self.reader_opts.filename_filter]

    def _get_metadata(self, filenames):

        if self.reader_opts.filename_filter is None:
            return self._all_metadata

        return [metadata for metadata in self._all_metadata if metadata['filename'] in filenames]

    @property
    def filenames(self):
        return self._get_filenames()

    @property
    def metadata(self):
        return self._get_metadata(strip_paths(self._get_filenames()))

    @property
    def document_index(self) -> pd.DataFrame:

        if self.reader_opts.filename_filter is None:
            return self._document_index

        return self._document_index[self._document_index.filename.isin(self.reader_opts.filename_filter)]

    def apply_filter(self, filename_filter: List[str]):
        self.reader_opts.filename_filter = filename_filter

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
