from __future__ import annotations

from typing import TYPE_CHECKING

from ..readers.tokenize_reader import TokenizeTextReader
from ..sparv.sparv_csv_to_text import SparvCsvToText

if TYPE_CHECKING:
    from ..readers import ExtractTaggedTokensOpts, TextReaderOpts, TextSource

# pylint: disable=too-many-arguments, super-with-arguments


class SparvCsvReader(TokenizeTextReader):
    def __init__(
        self,
        source: TextSource,
        reader_opts: TextReaderOpts,
        extract_opts: ExtractTaggedTokensOpts,
        *,
        chunk_size: int = None,
    ):
        self.delimiter: str = '\t'
        super().__init__(
            source,
            reader_opts=reader_opts.copy(filename_pattern='*.csv'),
            transform_opts=None,
            tokenize=lambda x: x.split(),
            chunk_size=chunk_size,
        )

        self.extract_tokens_opts = extract_opts
        self.parser = SparvCsvToText(delimiter=self.delimiter, extract_tokens_opts=self.extract_tokens_opts)

    def preprocess(self, content):
        return self.parser.transform(content)
