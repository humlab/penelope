# -*- coding: utf-8 -*-
import logging
import os
from typing import Callable, Iterable, List, Tuple, Union

from nltk.tokenize import word_tokenize

import penelope.utility.file_utility as file_utility
from penelope.corpus import readers
from penelope.corpus.text_transformer import TRANSFORMS, TextTransformer

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments,too-many-instance-attributes


def strip_path_and_extension(filename):

    return os.path.splitext(os.path.basename(filename))[0]


def strip_path_and_add_counter(filename, n_chunk):

    return '{}_{}.txt'.format(os.path.basename(filename), str(n_chunk).zfill(3))


class TextTokenizer:
    """Reads a text corpus from `source` and applies given transforms.
    Derived classes can override `preprocess` as an initial step before transforms are applied.
    The `preprocess` is applied on the entire document, and the transforms on each token.
    The `preprocess can for instance be used to extract text from an XML file (see derived class SParvXmlCorpusSourceReader)
    """

    def __init__(
        self,
        source_path=None,
        transforms=None,
        chunk_size=None,
        filename_pattern=None,
        filename_filter: Union[Callable, List[str]] = None,
        tokenize=None,
        as_binary=False,
        fix_whitespaces: bool = False,
        fix_hyphenation: bool = False,
        filename_fields=None,
    ):
        """
        Parameters
        ----------
        source : str
            Source can be either a Zip archive, a single text file or a directory.
        transforms : fn, optional
            List of transforms that in sequence are applied to each token
        chunk_size : int, optional
            Optional chunking in equal sizes of each document, by default None
        pattern : str, optional
            Pattern of files to include in the reprocess, by default None
        tokenize : Func[], optional
            Text tokenize function, by default None
        """
        self.source = readers.streamify_text_source(
            source_path, filename_pattern=filename_pattern, filename_filter=filename_filter, as_binary=as_binary
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

        self.filenames = file_utility.list_filenames(
            source_path, filename_pattern=filename_pattern, filename_filter=filename_filter
        )
        self.basenames = [os.path.basename(filename) for filename in self.filenames]
        self.metadata = [file_utility.extract_filename_fields(x, **(filename_fields or dict())) for x in self.basenames]
        self.metadict = {x.filename: x for x in (self.metadata or [])}

    def _create_iterator(self):
        return (
            (os.path.basename(document_name), document)
            for (filename, content) in self.source
            for document_name, document in self.process(filename, content)
        )

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
