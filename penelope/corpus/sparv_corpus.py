from __future__ import annotations

import penelope.corpus.readers as readers
import penelope.corpus.tokenized_corpus as tokenized_corpus
import penelope.utility.file_utility as file_utility
from penelope.corpus.tokens_transformer import transformer_defaults


class SparvTokenizedXmlCorpus(tokenized_corpus.TokenizedCorpus):
    def __init__(
        self,
        source,
        version,
        *,
        pos_includes=None,
        pos_excludes="|MAD|MID|PAD|",
        lemmatize=True,
        chunk_size=None,
        **tokens_transform_opts,
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
            [description], by default True
        chunk_size : [type], optional
            [description], by default None
        """
        tokens_transform_opts = {k: v for k, v in tokens_transform_opts.items() if k in transformer_defaults()}

        tokenizer = readers.SparvXmlTokenizer(
            source,
            transforms=None,
            pos_includes=pos_includes,
            pos_excludes=pos_excludes,
            lemmatize=lemmatize,
            chunk_size=chunk_size,
            xslt_filename=None,
            append_pos="",
            version=version,
        )
        super().__init__(tokenizer, **tokens_transform_opts)



class SparvTokenizedCsvCorpus(tokenized_corpus.TokenizedCorpus):
    """A tokenized Corpus for Sparv CSV files

    Parameters
    ----------
    tokenized_corpus : [type]
        [description]
    """
    def __init__(
        self,
        source,
        *,
        pos_includes: str=None,
        pos_excludes: str="|MAD|MID|PAD|",
        lemmatize: bool=True,
        chunk_size: int=None,
        append_pos: bool=False,
        **tokens_transform_opts,
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
        chunk_size : int, optional
            [description], by default None
        append_pos : bool, optional
            [description], by default False
        """

        tokens_transform_opts = {k: v for k, v in tokens_transform_opts.items() if k in transformer_defaults()}

        tokenizer = readers.SparvCsvTokenizer(
            source,
            transforms=None,
            pos_includes=pos_includes,
            pos_excludes=pos_excludes,
            lemmatize=lemmatize,
            chunk_size=chunk_size,
            append_pos=append_pos,
        )
        super().__init__(tokenizer, **tokens_transform_opts)


def sparv_xml_extract_and_store(source: str, target: str, version: int, **opts):
    """[summary]

    Parameters
    ----------
    source : str
        [description]
    target : str
        [description]
    version : int
        [description]
    """
    corpus = SparvTokenizedXmlCorpus(source, version, **opts)

    file_utility.store(target, corpus)

def sparv_csv_extract_and_store(source: str, target: str, **opts):
    """Extracts and stores text documents from a Sparv corpus in CSV format

    Parameters
    ----------
    source : str
        Corpus source reader
    target : str
        [description]
    version : int
        [description]
    """
    corpus = SparvTokenizedCsvCorpus(source, **opts)

    file_utility.store(target, corpus)

