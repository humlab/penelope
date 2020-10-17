# -*- coding: utf-8 -*-
import logging

from penelope.corpus.sparv.sparv_csv_to_text import SparvCsvToText

from .interfaces import FilenameOrFolderOrZipOrList
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
        source: FilenameOrFolderOrZipOrList,
        *,
        pos_includes: str = None,
        pos_excludes: str = "|MAD|MID|PAD|",
        lemmatize: bool = True,
        append_pos: bool = "",
        **tokenizer_opts,
    ):
        """[summary]

        Parameters
        ----------
        source : [type]
            [description]
        pos_includes : str, optional
            [description], by default None
        pos_excludes : str, optional
            [description], by default "|MAD|MID|PAD|"
        lemmatize : bool, optional
            [description], by default True
        append_pos : bool, optional
            [description], by default ""
        tokenizer_opts : Dict[str, Any]
            chunk_size : int
                Optional chunking of text in chunk_size pieces
            filename_pattern : str
                Filename pattern
            filename_filter: Union[Callable, List[str]]
                Filename inclusion predicate filter, or list of filenames to include
            filename_fields : Dict[str,Union[Callable,str]]
                Document metadata fields to extract from filename
            as_binary : bool
                Open input file as binary file (XML)

        """
        self.delimiter: str = '\t'

        super().__init__(
            source,
            **{**dict(tokenize=lambda x: x.split(), filename_pattern='*.csv', transforms=None), **tokenizer_opts},
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
