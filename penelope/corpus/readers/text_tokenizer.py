# -*- coding: utf-8 -*-
import logging
import os
from typing import Callable, Iterable, List, Tuple, Union

from nltk.tokenize import word_tokenize

import penelope.utility.file_utility as file_utility
from penelope.corpus.text_transformer import TRANSFORMS, TextTransformer

from .interfaces import FilenameOrFolderOrZipOrList, ICorpusReader
from .streamify_text_source import streamify_text_source

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments,too-many-instance-attributes


def strip_path_and_extension(filename):

    return os.path.splitext(os.path.basename(filename))[0]


def strip_path_and_add_counter(filename, n_chunk):

    return '{}_{}.txt'.format(os.path.basename(filename), str(n_chunk).zfill(3))


# class TextTokenizer(collections.abc.Iterable[Tuple[str,List[str]]]):
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
        filename_filter: Union[Callable, List[str]] = None,
        filename_fields=None,
        tokenize: Callable = None,
        fix_whitespaces: bool = False,
        fix_hyphenation: bool = False,
        as_binary: bool = False,
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
        filename_fields : [type], optional
            [description], by default None
        tokenize : Callable, optional
            [description], by default None
        fix_whitespaces : bool, optional
            [description], by default False
        fix_hyphenation : bool, optional
            [description], by default False
        as_binary : bool, optional
            [description], by default False
        """
        self.source = streamify_text_source(
            source, filename_pattern=filename_pattern, filename_filter=filename_filter, as_binary=as_binary
        )
        self.chunk_size = chunk_size
        self.tokenize = tokenize or word_tokenize

        self.text_transformer = (
            TextTransformer(transforms=transforms)
            .add(TRANSFORMS.fix_unicode)
            .add(TRANSFORMS.fix_whitespaces, condition=fix_whitespaces)
            .add(TRANSFORMS.fix_hyphenation, condition=fix_hyphenation)
        )

        self.iterator = None

        self._filenames = file_utility.list_filenames(
            source, filename_pattern=filename_pattern, filename_filter=filename_filter
        )
        self._basenames = [os.path.basename(filename) for filename in self._filenames]
        self._metadata = [
            file_utility.extract_filename_fields(x, **(filename_fields or dict())) for x in self._basenames
        ]

    def _create_iterator(self):
        return (
            (os.path.basename(document_name), document)
            for (filename, content) in self.source
            for document_name, document in self.process(filename, content)
        )

    @property
    def filenames(self):
        return self._filenames

    @property
    def metadata(self):
        return self._metadata

    @property
    def metalookup(self):
        return {x['filename']: x for x in (self._metadata or [])}

    def preprocess(self, content: str) -> str:
        """Process of source text that happens before any tokenization e.g. XML to text transform """
        return content

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
        tokens = self.tokenize(text)

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
        if self.iterator is None:
            self.iterator = self._create_iterator()
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise
