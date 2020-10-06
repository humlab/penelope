# -*- coding: utf-8 -*-
import logging
from typing import Callable, List

from penelope.corpus.sparv.sparv_xml_to_text import (XSLT_FILENAME_V3,
                                                     SparvXml2Text)

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


class SparvXmlTokenizer(TextTokenizer):
    def __init__(
        self,
        source,
        transforms: List[Callable] = None,
        pos_includes: str = None,
        pos_excludes: str = "|MAD|MID|PAD|",
        lemmatize: bool = True,
        chunk_size: int = None,
        xslt_filename: str = None,
        append_pos: bool = "",
        version: int = 4,
    ):

        self.delimiter: str = ' '
        tokenize = lambda x: x.split(self.delimiter)

        super().__init__(source, transforms, chunk_size, filename_pattern='*.xml', tokenize=tokenize, as_binary=True)

        self.lemmatize = lemmatize
        self.append_pos = append_pos
        self.pos_includes = pos_includes
        self.pos_excludes = pos_excludes
        self.xslt_filename = XSLT_FILENAME_V3 if version == 3 else xslt_filename
        self.parser = SparvXml2Text(
            xslt_filename=self.xslt_filename,
            delimiter=self.delimiter,
            pos_includes=self.pos_includes,
            lemmatize=self.lemmatize,
            append_pos=self.append_pos,
            pos_excludes=self.pos_excludes,
        )

    def preprocess(self, content):
        return self.parser.transform(content)


class Sparv3XmlTokenizer(SparvXmlTokenizer):
    def __init__(
        self,
        source,
        transforms: List[Callable] = None,
        pos_includes: str = None,
        pos_excludes: str = "|MAD|MID|PAD|",
        lemmatize: str = True,
        chunk_size: int = None,
        append_pos: str = "",
    ):

        super().__init__(
            source,
            transforms=transforms,
            pos_includes=pos_includes,
            lemmatize=lemmatize,
            chunk_size=chunk_size,
            xslt_filename=XSLT_FILENAME_V3,
            append_pos=append_pos,
            pos_excludes=pos_excludes,
        )
