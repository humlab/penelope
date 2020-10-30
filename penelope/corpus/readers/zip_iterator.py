# -*- coding: utf-8 -*-
import logging
from typing import Callable, List, Sequence, Union

from penelope.corpus.readers import ICorpusReader
from penelope.utility import IndexOfSplitOrCallableOrRegExp, create_iterator, extract_filenames_fields, list_filenames

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class ZipTextIterator(ICorpusReader):
    """Iterator that returns filename and content for each matching file in archive."""

    def __init__(
        self,
        source_path: str,
        *,
        filename_pattern: str = "*.txt",
        filename_filter: Union[List[str], Callable] = None,
        filename_fields: Sequence[IndexOfSplitOrCallableOrRegExp] = None,
        as_binary: bool = False,
    ):
        """Iterates a text corpus stored as textfiles in a zip archive
        Parameters
        ----------
        source_path : str
            [description]
        filename_pattern : str
            [description]
        filename_filter : List[str], optional
            [description], by default None
        filename_fields: Sequence[file_utility.IndexOfSplitOrCallableOrRegExp],
            [description], by default None
        as_binary : bool, optional
            If true then files are opened as `rb` and no decoding, by default False
        """
        self.source_path = source_path
        self._filenames = list_filenames(
            source_path, filename_pattern=filename_pattern, filename_filter=filename_filter
        )
        self.filename_fields = filename_fields
        self._metadata = None
        self.as_binary = as_binary
        self.iterator = None

    def _create_metadata(self):
        return extract_filenames_fields(filenames=self.filenames, filename_fields=self.filename_fields)

    def _create_iterator(self):
        return create_iterator(self.source_path, filenames=self.filenames, as_binary=self.as_binary)

    @property
    def filenames(self):
        return self._filenames

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self._create_metadata()
        return self._metadata

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = self._create_iterator()
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise
