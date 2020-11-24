# -*- coding: utf-8 -*-
import logging

from penelope.corpus.sparv.sparv_csv_to_text import SparvCsvToText

from .interfaces import ExtractTokensOpts, TextSource
from .text_tokenizer import TextTokenizer

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, super-with-arguments


class SparvCsvTokenizer(TextTokenizer):
    def __init__(
        self,
        source: TextSource,
        *,
        extract_tokens_opts: ExtractTokensOpts = None,
        **tokenizer_opts,
    ):
        """[summary]

        Parameters
        ----------
        source : [type]
            [description]
        extract_tokens_opts : ExtractTokensOpts, optional
        tokenizer_opts : Dict[str, Any]
            Optional chunking of text in chunk_size pieces
            filename_pattern : str
            Filename pattern
            filename_filter: Union[Callable, List[str]]
                Filename inclusion predicate filter, or list of filenames to include
            filename_fields : Sequence[Sequence[IndexOfSplitOrCallableOrRegExp]]
                Document metadata fields to extract from filename
            as_binary : bool
                Open input file as binary file (XML)

        """
        self.delimiter: str = '\t'

        super().__init__(
            source,
            **{**dict(tokenize=lambda x: x.split(), filename_pattern='*.csv'), **tokenizer_opts},
        )

        self.extract_tokens_opts = extract_tokens_opts or ExtractTokensOpts()
        self.parser = SparvCsvToText(delimiter=self.delimiter, extract_tokens_opts=self.extract_tokens_opts)

    def preprocess(self, content):
        return self.parser.transform(content)
