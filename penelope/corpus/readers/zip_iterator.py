# -*- coding: utf-8 -*-
import logging

from penelope.utility import create_iterator, extract_filenames_metadata, list_filenames

from ..document_index import DocumentIndex, metadata_to_document_index
from .interfaces import ICorpusReader
from .text_reader import TextReaderOpts

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class ZipTextIterator(ICorpusReader):
    """Iterator that returns filename and content for each matching file in archive."""

    def __init__(self, source_path: str, reader_opts: TextReaderOpts):
        """Iterates a text corpus stored as textfiles in a zip archive
        Parameters
        ----------
        source_path : str
            [description]
        reader_opts : TextReaderOpts
            [description]
        """
        self.reader_opts = reader_opts
        self.source_path = source_path
        self._filenames = list_filenames(
            source_path,
            filename_pattern=self.reader_opts.filename_pattern,
            filename_filter=self.reader_opts.filename_filter,
        )
        self._metadata = None
        self.iterator = None

    def _create_metadata(self):
        return extract_filenames_metadata(filenames=self.filenames, filename_fields=self.reader_opts.filename_fields)

    def _create_iterator(self):
        return create_iterator(self.source_path, filenames=self.filenames, as_binary=self.reader_opts.as_binary)

    @property
    def filenames(self):
        return self._filenames

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self._create_metadata()
        return self._metadata

    @property
    def document_index(self) -> DocumentIndex:
        return metadata_to_document_index(self.metadata, document_id_field=self.reader_opts.index_field)

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
