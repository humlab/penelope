from __future__ import annotations

from typing import TYPE_CHECKING

import penelope.utility.zip_utils as zip_util

from ..readers.interfaces import ExtractTaggedTokensOpts, TextReaderOpts
from ..tokenized_corpus import TokenizedCorpus
from .sparv_csv_reader import SparvCsvReader
from .sparv_xml_reader import SparvXmlReader

if TYPE_CHECKING:
    from ..transform import TokensTransformOpts


class SparvTokenizedXmlCorpus(TokenizedCorpus):
    def __init__(
        self,
        source,
        version,
        *,
        reader_opts: TextReaderOpts,
        extract_opts: ExtractTaggedTokensOpts,
        transform_opts: TokensTransformOpts,
        chunk_size: int = None,
    ):
        if isinstance(source, SparvXmlReader):
            tokens_reader = source
        else:
            tokens_reader = SparvXmlReader(
                source,
                extract_opts=extract_opts,
                xslt_filename=None,
                version=version,
                reader_opts=reader_opts,
                chunk_size=chunk_size,
            )

        super().__init__(tokens_reader, transform_opts=transform_opts)


class SparvTokenizedCsvCorpus(TokenizedCorpus):
    """A tokenized Corpus for Sparv CSV files

    Parameters
    ----------
    tokenized_corpus : [type]

    """

    def __init__(
        self,
        source,
        *,
        reader_opts: TextReaderOpts,
        extract_opts: ExtractTaggedTokensOpts,
        transform_opts: TokensTransformOpts,
        chunk_size: int = None,
    ):
        if isinstance(source, SparvCsvReader):
            tokens_reader = source
        else:
            tokens_reader = SparvCsvReader(
                source,
                extract_opts=extract_opts,
                reader_opts=reader_opts,
                chunk_size=chunk_size,
            )
        super().__init__(tokens_reader, transform_opts=transform_opts)


def sparv_xml_extract_and_store(
    source: str,
    target: str,
    version: int,
    extract_opts: ExtractTaggedTokensOpts,
    reader_opts: TextReaderOpts,
    transform_opts: TokensTransformOpts,
    chunk_size: int = None,
):
    """[summary]

    Parameters
    ----------
    source : str
    target : str
    version : int
    extract_tokens_opts : ExtractTaggedTokensOpts, optional
    transform_opts : TokensTransformOpts, optional
        only_alphabetic: bool = False,
        only_any_alphanumeric: bool = False,
        to_lower: bool = False,
        to_upper: bool = False,
        min_len: int = None,
        max_len: int = None,
        remove_accents: bool = False,
        remove_stopwords: str = None,
        extra_stopwords: List[str] = None,
        remove_numerals: bool = False,
        remove_symbols: bool = False,
    reader_opts : TextReaderOpts
        Passed to source reader:
    chunk_size: int = None,
    """
    corpus = SparvTokenizedXmlCorpus(
        source,
        version,
        extract_opts=extract_opts,
        reader_opts=reader_opts,
        transform_opts=transform_opts,
        chunk_size=chunk_size,
    )

    zip_util.store(zip_or_filename=target, stream=corpus)


def sparv_csv_extract_and_store(
    source: str,
    target: str,
    extract_opts: ExtractTaggedTokensOpts = None,
    reader_opts: TextReaderOpts = None,
    transform_opts: TokensTransformOpts = None,
    chunk_size: int = None,
):
    """Extracts and stores text documents from a Sparv corpus in CSV format

    Parameters
    ----------
    source : str
        [description]
    target : str
        [description]
    extract_opts : ExtractTaggedTokensOpts, optional
    transform_opts : TokensTransformOpts, optional
        only_alphabetic: bool = False,
        only_any_alphanumeric: bool = False,
        to_lower: bool = False,
        to_upper: bool = False,
        min_len: int = None,
        max_len: int = None,
        remove_accents: bool = False,
        remove_stopwords: str = None,
        extra_stopwords: List[str] = None,
        remove_numerals: bool = False,
        keep_symbols: bool = True,
    reader_opts : Dict[str, Any], optional
    """
    corpus = SparvTokenizedCsvCorpus(
        source,
        extract_opts=extract_opts,
        reader_opts=reader_opts,
        transform_opts=transform_opts,
        chunk_size=chunk_size,
    )

    zip_util.store(zip_or_filename=target, stream=corpus)
