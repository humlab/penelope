import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

from penelope.utility import (
    IndexOfSplitOrCallableOrRegExp,
    extract_filenames_fields,
    filename_satisfied_by,
    list_filenames,
    strip_path_and_extension,
    strip_paths,
)
from penelope.vendor.nltk import word_tokenize

from .interfaces import FilenameOrFolderOrZipOrList, ICorpusReader
from .streamify_text_source import streamify_text_source
from .text_transformer import TextTransformer, TextTransformOpts

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments,too-many-instance-attributes

FilenameOrCallableOrSequenceFilter = Union[Callable, Sequence[str]]


class TextTokenizer(ICorpusReader):
    """Reads a text corpus from `source` and applies given transforms.
    Derived classes can override `preprocess` as an initial step before transforms are applied.
    The `preprocess` is applied on the entire document, and the transforms on each token.
    The `preprocess can for instance be used to extract text from an XML file (see derived class SParvXmlCorpusSourceReader)
    """

    def __init__(
        self,
        source: FilenameOrFolderOrZipOrList,
        *,
        transforms: List[Callable] = None,
        chunk_size: int = None,
        filename_pattern: str = None,
        filename_filter: FilenameOrCallableOrSequenceFilter = None,
        filename_fields: Sequence[IndexOfSplitOrCallableOrRegExp] = None,
        tokenize: Callable = None,
        as_binary: bool = False,
        text_transform_opts: TextTransformOpts = None,
    ):
        """[summary]

        Parameters
        ----------
        source : FilenameOrFolderOrZipOrList
            [description]
        transforms : List[Callable], optional
            [description], by default None
        chunk_size : int, optional
            [description], by default None
        filename_pattern : str, optional
            [description], by default None
        filename_filter : Union[Callable, List[str]], optional
            [description], by default None
        filename_fields : Sequence[IndexOfSplitOrCallableOrRegExp], optional
            [description], by default None
        tokenize : Callable, optional
            [description], by default None
        as_binary : bool, optional
            [description], by default False
        text_transform_opts : TextTransformOpts
        """
        self._source = source
        self._as_binary = as_binary
        self._filename_filter = filename_filter
        self._filename_fields = filename_fields
        self._filename_pattern = filename_pattern
        self._tokenize = tokenize or word_tokenize
        self.chunk_size = chunk_size

        text_transform_opts = text_transform_opts or TextTransformOpts()
        self.text_transformer = TextTransformer(transforms=transforms, text_transform_opts=text_transform_opts)

        self._iterator = None

        self._all_filenames = list_filenames(source, filename_pattern=filename_pattern, filename_filter=None)

        self._all_metadata = self._create_all_metadata()

    def _get_texts(self):
        return streamify_text_source(
            self._source,
            filename_pattern=self._filename_pattern,
            filename_filter=self._filename_filter,
            as_binary=self._as_binary,
        )

    def _create_iterator(self):
        return (
            (os.path.basename(filename), document)
            for (filename, content) in self._get_texts()
            for filename, document in self.process(filename, content)
        )

    def _create_all_metadata(self) -> Sequence[Dict[str, Any]]:
        return extract_filenames_fields(filenames=self._all_filenames, filename_fields=self._filename_fields)

    def _get_filenames(self):

        if self._filename_filter is None:
            return self._all_filenames

        return [
            filename
            for filename in self._all_filenames
            if filename_satisfied_by(filename, filename_pattern=None, filename_filter=self._filename_filter)
        ]

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
    def metadata_lookup(self):
        return {x['filename']: x for x in self.metadata}

    def preprocess(self, content: str) -> str:
        """Process of source text that happens before any tokenization e.g. XML to text transform """
        return content

    def apply_filter(self, filename_filter: FilenameOrCallableOrSequenceFilter):
        self._filename_filter = filename_filter

    def process(self, filename: str, content: str) -> Iterable[Tuple[str, List[str]]]:
        """Process a document and returns tokenized text, and optionally splits text in equal length chunks

        Parameters
        ----------
        content : str
            The actual text read from source.

        Yields
        -------
        Tuple[str,List[str]]
            Filename and tokens
        """
        text = self.preprocess(content)
        text = self.text_transformer.transform(text)
        tokens = self._tokenize(text)

        if self.chunk_size is None:

            stored_name = strip_path_and_extension(filename) + '.txt'

            yield stored_name, tokens

        else:

            tokens = list(tokens)

            for n_chunk, i in enumerate(range(0, len(tokens), self.chunk_size)):

                stored_name = '{}_{}.txt'.format(strip_path_and_extension(filename), str(n_chunk + 1).zfill(3))

                yield stored_name, tokens[i : i + self.chunk_size]

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
