from __future__ import annotations

from typing import Any, Dict

from penelope.utility import store_to_archive

from . import readers
from .readers import AnnotationOpts
from .tokenized_corpus import TokenizedCorpus
from .tokens_transformer import TokensTransformOpts


class SparvTokenizedXmlCorpus(TokenizedCorpus):
    def __init__(
        self,
        source,
        version,
        *,
        annotation_opts: AnnotationOpts = None,
        tokens_transform_opts: TokensTransformOpts = None,
        tokenizer_opts: Dict[str, Any] = None,
    ):
        """[summary]

        Parameters
        ----------
        source : [type]
            [description]
        version : [type]
            [description]
        annotation_opts : AnnotationOpts, optional
            [description], by default None
        tokens_transform_opts :TokensTransformOpts, optional
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
        tokenizer_opts : Dict[str, Any], optional
            Passed to source reader:
                transforms: List[Callable] = None,
                text_transforms_opts: TextTransformOpts
                chunk_size: int = None,
                filename_pattern: str = None,
                filename_filter: Union[Callable, List[str]] = None,
                filename_fields=None,
                N/A: tokenize: Callable = None,
                as_binary: bool = False,
        """

        if isinstance(source, readers.SparvXmlTokenizer):
            tokenizer = source
        else:
            tokenizer = readers.SparvXmlTokenizer(
                source,
                annotation_opts=annotation_opts or AnnotationOpts(),
                xslt_filename=None,
                version=version,
                **(tokenizer_opts or {}),
            )

        super().__init__(tokenizer, tokens_transform_opts=tokens_transform_opts)


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
        annotation_opts: AnnotationOpts = None,
        tokens_transform_opts: TokensTransformOpts = None,
        tokenizer_opts: Dict[str, Any] = None,
    ):
        if isinstance(source, readers.SparvCsvTokenizer):
            tokenizer = source
        else:
            tokenizer = readers.SparvCsvTokenizer(
                source,
                annotation_opts=annotation_opts,
                **(tokenizer_opts or {}),
            )
        super().__init__(tokenizer, tokens_transform_opts=tokens_transform_opts)


def sparv_xml_extract_and_store(
    source: str,
    target: str,
    version: int,
    annotation_opts: AnnotationOpts = None,
    tokenizer_opts=None,
    tokens_transform_opts: TokensTransformOpts = None,
):
    """[summary]

    Parameters
    ----------
    source : str
    target : str
    version : int
    annotation_opts : AnnotationOpts, optional
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
    tokenizer_opts : Dict[str, Any], optional
        Passed to source reader:
            transforms: List[Callable] = None,
            text_transforms_opts: TextTransformOpts
            chunk_size: int = None,
            filename_pattern: str = None,
            filename_filter: Union[Callable, List[str]] = None,
            filename_fields=None,
            N/A: tokenize: Callable = None,
            as_binary: bool = False,
    """
    corpus = SparvTokenizedXmlCorpus(
        source,
        version,
        annotation_opts=annotation_opts,
        tokenizer_opts=tokenizer_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    store_to_archive(target, corpus)


def sparv_csv_extract_and_store(
    source: str,
    target: str,
    annotation_opts: AnnotationOpts = None,
    tokenizer_opts=None,
    tokens_transform_opts: TokensTransformOpts = None,
):
    """Extracts and stores text documents from a Sparv corpus in CSV format

    Parameters
    ----------
    source : str
        [description]
    target : str
        [description]
    annotation_opts : AnnotationOpts, optional
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
    tokenizer_opts : Dict[str, Any], optional
        Passed to source reader:
            transforms: List[Callable] = None,
            text_transforms_opts: TextTransformOpts
            chunk_size: int = None,
            filename_pattern: str = None,
            filename_filter: Union[Callable, List[str]] = None,
            filename_fields=None,
            N/A: tokenize: Callable = None,
            as_binary: bool = False,
    """
    corpus = SparvTokenizedCsvCorpus(
        source,
        annotation_opts=annotation_opts,
        tokenizer_opts=tokenizer_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    store_to_archive(target, corpus)
