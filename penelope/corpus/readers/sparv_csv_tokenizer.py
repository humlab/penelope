# -*- coding: utf-8 -*-
import logging
from typing import Any, Callable, Dict, List

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


# source_path=None,
# transforms=None,
# chunk_size=None,
# filename_pattern=None,
# filename_filter: Union[Callable, List[str]] = None,
# tokenize=None,
# as_binary=False,
# fix_whitespaces: bool = False,
# fix_hyphenation: bool = False,
# filename_fields=None,
class SparvCsvTokenizer(TextTokenizer):

    def __init__(
        self,
        source,
        pos_includes: str = None,
        pos_excludes: str = "|MAD|MID|PAD|",
        lemmatize: bool = True,
        append_pos: bool = "",
        **tokenizer_opts
    ):

        self.delimiter: str = '\t'

        super().__init__(
            source, **{
                **dict(tokenize=lambda x: x.split(), filename_pattern='*.csv', transforms=None),
                **tokenizer_opts
            }
        )

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
