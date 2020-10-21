from __future__ import annotations

from typing import Any, Dict

import penelope.corpus.readers as readers
import penelope.corpus.tokenized_corpus as tokenized_corpus
import penelope.utility.file_utility as file_utility
from penelope.corpus.tokens_transformer import transformer_defaults_filter


class SparvTokenizedXmlCorpus(tokenized_corpus.TokenizedCorpus):
    def __init__(
        self,
        source,
        version,
        *,
        pos_includes=None,
        pos_excludes="|MAD|MID|PAD|",
        lemmatize=True,
        tokens_transform_opts: Dict[str, Any] = None,
        tokenizer_opts: Dict[str, Any] = None,
    ):
        """[summary]

        Parameters
        ----------
        source : [type]
            [description]
        version : [type]
            [description]
        pos_includes : [type], optional
            [description], by default None
        pos_excludes : str, optional
            [description], by default "|MAD|MID|PAD|"
        lemmatize : bool, optional
            If True then use token baseform, by default True
        tokens_transform_opts : Dict[str, Any], optional
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
                chunk_size: int = None,
                filename_pattern: str = None,
                filename_filter: Union[Callable, List[str]] = None,
                filename_fields=None,
                N/A: tokenize: Callable = None,
                N/A: fix_whitespaces: bool = False,
                N/A: fix_hyphenation: bool = False,
                as_binary: bool = False,
        """

        if isinstance(source, readers.SparvXmlTokenizer):
            tokenizer = source
        else:
            tokenizer = readers.SparvXmlTokenizer(
                source,
                pos_includes=pos_includes,
                pos_excludes=pos_excludes,
                xslt_filename=None,
                append_pos="",
                version=version,
                **{'lemmatize': lemmatize, **(tokenizer_opts or {})},
            )

        super().__init__(tokenizer, **transformer_defaults_filter(tokens_transform_opts))


class SparvTokenizedCsvCorpus(tokenized_corpus.TokenizedCorpus):
    """A tokenized Corpus for Sparv CSV files

    Parameters
    ----------
    tokenized_corpus : [type]

    """

    def __init__(
        self,
        source,
        *,
        pos_includes: str = None,
        pos_excludes: str = "|MAD|MID|PAD|",
        lemmatize: bool = True,
        append_pos: bool = False,
        tokens_transform_opts: Dict[str, Any] = None,
        tokenizer_opts: Dict[str, Any] = None,
    ):
        if isinstance(source, readers.SparvCsvTokenizer):
            tokenizer = source
        else:
            tokenizer = readers.SparvCsvTokenizer(
                source,
                pos_includes=pos_includes,
                pos_excludes=pos_excludes,
                append_pos=append_pos,
                **{'lemmatize': lemmatize, **(tokenizer_opts or {})},
            )
        super().__init__(tokenizer, **transformer_defaults_filter(tokens_transform_opts))


def sparv_xml_extract_and_store(
    source: str,
    target: str,
    version: int,
    pos_includes: str = None,
    pos_excludes: str = "|MAD|MID|PAD|",
    lemmatize: bool = True,
    # append_pos: bool = False,
    tokenizer_opts=None,
    tokens_transform_opts=None,
):
    """[summary]

    Parameters
    ----------
    source : str
        [description]
    target : str
        [description]
    version : int
        [description]
    pos_includes : [type], optional
        [description], by default None
    pos_excludes : str, optional
        [description], by default "|MAD|MID|PAD|"
    lemmatize : bool, optional
        If True then use token baseform, by default True
    tokens_transform_opts : Dict[str, Any], optional
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
            chunk_size: int = None,
            filename_pattern: str = None,
            filename_filter: Union[Callable, List[str]] = None,
            filename_fields=None,
            N/A: tokenize: Callable = None,
            N/A: fix_whitespaces: bool = False,
            N/A: fix_hyphenation: bool = False,
            as_binary: bool = False,
    """
    corpus = SparvTokenizedXmlCorpus(
        source,
        version,
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        lemmatize=lemmatize,
        # append_pos=append_pos, FIXA
        tokenizer_opts=tokenizer_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    file_utility.store(target, corpus)


def sparv_csv_extract_and_store(
    source: str,
    target: str,
    pos_includes: str = None,
    pos_excludes: str = "|MAD|MID|PAD|",
    lemmatize: bool = True,
    append_pos: bool = False,
    tokenizer_opts=None,
    tokens_transform_opts=None,
):
    """Extracts and stores text documents from a Sparv corpus in CSV format

    Parameters
    ----------
    source : str
        [description]
    target : str
        [description]
    pos_includes : [type], optional
        [description], by default None
    pos_excludes : str, optional
        [description], by default "|MAD|MID|PAD|"
    lemmatize : bool, optional
        If True then use token baseform, by default True
    tokens_transform_opts : Dict[str, Any], optional
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
            chunk_size: int = None,
            filename_pattern: str = None,
            filename_filter: Union[Callable, List[str]] = None,
            filename_fields=None,
            N/A: tokenize: Callable = None,
            N/A: fix_whitespaces: bool = False,
            N/A: fix_hyphenation: bool = False,
            as_binary: bool = False,
    """
    corpus = SparvTokenizedCsvCorpus(
        source,
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        lemmatize=lemmatize,
        append_pos=append_pos,
        tokenizer_opts=tokenizer_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    file_utility.store(target, corpus)
