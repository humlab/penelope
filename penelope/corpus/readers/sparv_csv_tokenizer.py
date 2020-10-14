# -*- coding: utf-8 -*-
import logging
from typing import Callable, List

from penelope.corpus.sparv.sparv_csv_to_text import SparvCsvToText

from .text_tokenizer import TextTokenizer

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, super-with-arguments

DEFAULT_OPTS = dict(
    pos_includes='',
    lemmatize=True,
    chunk_size=None,
    xslt_filename=None,
    delimiter="|",
    append_pos="",
    pos_excludes="|MAD|MID|PAD|",
)


class SparvCsvTokenizer(TextTokenizer):
    def __init__(
        self,
        source,
        transforms: List[Callable] = None,
        pos_includes: str = None,
        pos_excludes: str = "|MAD|MID|PAD|",
        lemmatize: bool = True,
        chunk_size: int = None,
        append_pos: bool = "",
    ):

        self.delimiter: str = '\t'
        tokenize = lambda x: x.split()

        super().__init__(source, transforms, chunk_size, filename_pattern='*.csv', tokenize=tokenize)

        self.lemmatize = lemmatize
        self.append_pos = append_pos
        self.pos_includes = pos_includes
        self.pos_excludes = pos_excludes
        self.parser = SparvCsvToText(
            delimiter=self.delimiter,
            pos_includes=self.pos_includes,
            lemmatize=self.lemmatize,
            append_pos=self.append_pos,
            pos_excludes=self.pos_excludes,
        )

    def preprocess(self, content):
        return self.parser.transform(content)
