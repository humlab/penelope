import logging
import os
from typing import Any, Dict, Iterable, Sequence, Tuple

import pandas as pd
from penelope.utility import (
    extract_filenames_metadata,
    filename_satisfied_by,
    list_filenames,
    metadata_to_document_index,
    strip_paths,
)

from .interfaces import FilenameOrCallableOrSequenceFilter, ICorpusReader, TextReaderOpts, TextSource
from .streamify_text_source import streamify_text_source
from .text_transformer import TextTransformer, TextTransformOpts

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments,too-many-instance-attributes


class TextReader(ICorpusReader):
    """Reads a text corpus from `source` and applies given transforms.
    Derived classes can override `preprocess` as an initial step before transforms are applied.
    The `preprocess can for instance be used to extract text from an XML file (see derived class SParvXmlCorpusSourceReader)
    """

    @staticmethod
    def create(
        source: TextSource, *, reader_opts: TextReaderOpts, transform_opts: TextTransformOpts = None
    ) -> "TextReader":
        return TextReader(source=source, reader_opts=reader_opts, transform_opts=transform_opts)

    def __init__(
        self,
        source: TextSource,
        *,
        reader_opts: TextReaderOpts = None,
        transform_opts: TextTransformOpts = None,
    ):
        reader_opts = reader_opts or TextReaderOpts()

        self._source = source
        self.reader_opts = reader_opts.copy()
        self.text_transformer = TextTransformer(text_transform_opts=transform_opts or TextTransformOpts())

        self._iterator = None
        self._all_filenames = list_filenames(
            source, filename_pattern=self.reader_opts.filename_pattern, filename_filter=None
        )
        self._all_metadata = self._create_all_metadata()

    def _get_texts(self):
        return streamify_text_source(
            self._source,
            filename_pattern=self.reader_opts.filename_pattern,
            filename_filter=self.reader_opts.filename_filter,
            as_binary=self.reader_opts.as_binary,
        )

    def _create_iterator(self):
        return (
            (os.path.basename(filename), document)
            for (filename, content) in self._get_texts()
            for filename, document in self.process(filename, content)
        )

    def _create_all_metadata(self) -> Sequence[Dict[str, Any]]:
        return extract_filenames_metadata(
            filenames=self._all_filenames,
            filename_fields=self.reader_opts.filename_fields,
        )

    def _get_filenames(self):

        if self.reader_opts.filename_filter is None:
            return self._all_filenames

        return [
            filename
            for filename in self._all_filenames
            if filename_satisfied_by(filename, filename_pattern=None, filename_filter=self.reader_opts.filename_filter)
        ]

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
    def metadata_lookup(self):
        return {x['filename']: x for x in self.metadata}

    @property
    def document_index(self) -> pd.DataFrame:
        _document_index: pd.DataFrame = metadata_to_document_index(self.metadata, self.reader_opts.filename_fields_key)
        return _document_index

    def preprocess(self, content: str) -> str:
        """Process of source text that happens before any tokenization e.g. XML to text transform """
        return content

    def apply_filter(self, filename_filter: FilenameOrCallableOrSequenceFilter):
        self.reader_opts.filename_filter = filename_filter

    def process(self, filename: str, content: str) -> Iterable[Tuple[str, str]]:  # pylint: disable=unused-argument
        """Process a document and yields tokenized text, and optionally splits text in equal length chunks
        Note: Is defomed a Iterable but only returns one item (consitency)
        Parameters
        ----------
        content : str
            The actual text read from source.

        Yields
        -------
        Tuple[str,str]
            Filename and tokens
        """
        text = self.preprocess(content)
        text = self.text_transformer.transform(text)
        yield filename, text

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
