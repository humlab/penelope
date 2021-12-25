from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple

import scipy.sparse as sp
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Sparse2Corpus, corpus2csc


def _id2token2token2id(id2token: Mapping[int, str]) -> dict:
    if id2token is None:
        return None
    if hasattr(id2token, 'token2id'):
        return id2token.token2id
    token2id: dict = {v: k for k, v in id2token.items()}
    return token2id


GensimBowCorpus = Iterable[Iterable[Tuple[int, float]]]


def from_stream_of_tokens_to_sparse2corpus(source: Any, vocabulary) -> Sparse2Corpus:
    bow_corpus: GensimBowCorpus = [vocabulary.doc2bow(tokens) for _, tokens in source]
    csc_matrix: sp.csc_matrix = corpus2csc(
        bow_corpus,
        num_terms=len(vocabulary),
        num_docs=len(bow_corpus),
        num_nnz=sum(map(len, bow_corpus)),
    )
    corpus: Sparse2Corpus = Sparse2Corpus(csc_matrix, documents_columns=True)
    return corpus


def from_stream_of_tokens_to_dictionary(source: Any, id2token: dict) -> Dictionary:
    """Creates a Dictionary from source using existing `id2token` mapping.
    Useful if cfs/dfs are needed, otherwise just use the existing mapping."""
    vocabulary: Dictionary = Dictionary()
    if id2token is not None:
        vocabulary.token2id = _id2token2token2id(id2token)
    vocabulary.add_documents(tokens for _, tokens in source)
    return vocabulary


def from_id2token_dict_to_dictionary(id2token: dict) -> Dictionary:
    """Creates a `Dictionary` from a id2word dict.
    TODO: Deprecate function. Not useful since the existing dict is sufficient in gensim"""
    if isinstance(id2token, Dictionary):
        return id2token

    if not isinstance(id2token, dict):
        raise ValueError(f"expected dict, found {type(id2token)}")

    dictionary: Dictionary = Dictionary()
    dictionary.id2token = id2token
    dictionary.token2id = dict((v, k) for v, k in id2token.items())

    return dictionary
