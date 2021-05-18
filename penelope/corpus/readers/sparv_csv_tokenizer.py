# -*- coding: utf-8 -*-
import logging

from ..sparv.sparv_csv_to_text import SparvCsvToText
from .interfaces import ExtractTaggedTokensOpts, TextReaderOpts, TextSource
from .text_tokenizer import TextTokenizer
from .text_transformer import TextTransformOpts

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, super-with-arguments


class SparvCsvTokenizer(TextTokenizer):
    def __init__(
        self,
        source: TextSource,
        reader_opts: TextReaderOpts,
        *,
        extract_opts: ExtractTaggedTokensOpts = None,
        chunk_size: int = None,
    ):
        """[summary]

        Parameters
        ----------
        source : [type]
            [description]
        extract_tokens_opts : ExtractTaggedTokensOpts, optional
        reader_opts : TextReaderOpts

        """
        self.delimiter: str = '\t'
        super().__init__(
            source,
            reader_opts=reader_opts.copy(filename_pattern='*.csv'),
            transform_opts=TextTransformOpts.empty(),
            tokenize=lambda x: x.split(),
            chunk_size=chunk_size,
        )

        self.extract_tokens_opts = extract_opts or ExtractTaggedTokensOpts(lemmatize=True)
        self.parser = SparvCsvToText(delimiter=self.delimiter, extract_tokens_opts=self.extract_tokens_opts)

    def preprocess(self, content):
        return self.parser.transform(content)
