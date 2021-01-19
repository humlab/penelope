from __future__ import annotations

import penelope.utility.zip_utils as zip_util

from . import readers
from .readers import ExtractTaggedTokensOpts, TextReaderOpts
from .tokenized_corpus import TokenizedCorpus
from .tokens_transformer import TokensTransformOpts


class SparvTokenizedXmlCorpus(TokenizedCorpus):
    def __init__(
        self,
        source,
        version,
        *,
        reader_opts: TextReaderOpts = None,
        extract_tokens_opts: ExtractTaggedTokensOpts = None,
        tokens_transform_opts: TokensTransformOpts = None,
        chunk_size: int = None,
    ):
        reader_opts = reader_opts or TextReaderOpts()

        if isinstance(source, readers.SparvXmlTokenizer):
            tokens_reader = source
        else:
            tokens_reader = readers.SparvXmlTokenizer(
                source,
                extract_tokens_opts=extract_tokens_opts or ExtractTaggedTokensOpts(lemmatize=True),
                xslt_filename=None,
                version=version,
                reader_opts=reader_opts,
                chunk_size=chunk_size,
            )

        super().__init__(tokens_reader, tokens_transform_opts=tokens_transform_opts)


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
        reader_opts: TextReaderOpts = None,
        extract_tokens_opts: ExtractTaggedTokensOpts = None,
        tokens_transform_opts: TokensTransformOpts = None,
        chunk_size: int = None,
    ):
        reader_opts = reader_opts or TextReaderOpts()
        if isinstance(source, readers.SparvCsvTokenizer):
            tokens_reader = source
        else:
            tokens_reader = readers.SparvCsvTokenizer(
                source,
                extract_tokens_opts=extract_tokens_opts,
                reader_opts=reader_opts,
                chunk_size=chunk_size,
            )
        super().__init__(tokens_reader, tokens_transform_opts=tokens_transform_opts)


def sparv_xml_extract_and_store(
    source: str,
    target: str,
    version: int,
    extract_tokens_opts: ExtractTaggedTokensOpts = None,
    reader_opts: TextReaderOpts = None,
    tokens_transform_opts: TokensTransformOpts = None,
    chunk_size: int = None,
):
    """[summary]

    Parameters
    ----------
    source : str
    target : str
    version : int
    extract_tokens_opts : ExtractTaggedTokensOpts, optional
    tokens_transform_opts : TokensTransformOpts, optional
        Passed to TokensTransformer:
            only_alphabetic: bool = False,
            only_any_alphanumeric: bool = False,
            to_lower: bool = False,
            to_upper: bool = False,
            min_len: int = None,
            max_len: int = None,
            remove_accents: bool = False,
            remove_stopwords: bool = False,
            stopwords: Any = None,
            extra_stopwords: List[str] = None,
            language: str = "swedish",
            keep_numerals: bool = True,
            keep_symbols: bool = True,
    reader_opts : TextReaderOpts
        Passed to source reader:
    chunk_size: int = None,
    """
    corpus = SparvTokenizedXmlCorpus(
        source,
        version,
        extract_tokens_opts=extract_tokens_opts,
        reader_opts=reader_opts,
        tokens_transform_opts=tokens_transform_opts,
        chunk_size=chunk_size,
    )

    zip_util.store(zip_or_filename=target, stream=corpus)


def sparv_csv_extract_and_store(
    source: str,
    target: str,
    extract_tokens_opts: ExtractTaggedTokensOpts = None,
    reader_opts: TextReaderOpts = None,
    tokens_transform_opts: TokensTransformOpts = None,
    chunk_size: int = None,
):
    """Extracts and stores text documents from a Sparv corpus in CSV format

    Parameters
    ----------
    source : str
        [description]
    target : str
        [description]
    extract_tokens_opts : ExtractTaggedTokensOpts, optional
    tokens_transform_opts : TokensTransformOpts, optional
        Passed to TokensTransformer:
            only_alphabetic: bool = False,
            only_any_alphanumeric: bool = False,
            to_lower: bool = False,
            to_upper: bool = False,
            min_len: int = None,
            max_len: int = None,
            remove_accents: bool = False,
            remove_stopwords: bool = False,
            stopwords: Any = None,
            extra_stopwords: List[str] = None,
            language: str = "swedish",
            keep_numerals: bool = True,
            keep_symbols: bool = True,
    reader_opts : Dict[str, Any], optional
    """
    corpus = SparvTokenizedCsvCorpus(
        source,
        extract_tokens_opts=extract_tokens_opts,
        reader_opts=reader_opts,
        tokens_transform_opts=tokens_transform_opts,
        chunk_size=chunk_size,
    )

    zip_util.store(zip_or_filename=target, stream=corpus)
