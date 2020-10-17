# -*- coding: utf-8 -*-
import logging
from typing import Callable, List, Union

import penelope.utility.file_utility as file_utility
from penelope.corpus.readers import ICorpusReader

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class ZipTextIterator(ICorpusReader):
    """Iterator that returns filename and content for each matching file in archive."""

    def __init__(
        self,
        source_path: str,
        filename_pattern: str = "*.txt",
        filename_filter: Union[List[str], Callable] = None,
        as_binary: bool = False,
    ):
        """
        Parameters
        ----------
        source_path : sttr
            [description]
        filename_pattern : str
            [description]
        filename_filter : List[str], optional
            [description], by default None
        as_binary : bool, optional
            If true then files are opened as `rb` and no decoding, by default False
        """
        self.source_path = source_path
        self._filenames = file_utility.list_filenames(
            source_path, filename_pattern=filename_pattern, filename_filter=filename_filter
        )
        self.as_binary = as_binary
        self.iterator = None

    def _create_iterator(self):
        return file_utility.create_iterator(self.source_path, filenames=self.filenames, as_binary=self.as_binary)

    @property
    def filenames(self):
        return self._filenames

    @property
    def metadata(self):
        return [{'filename': x for x in self._filenames}]

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
