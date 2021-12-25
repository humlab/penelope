from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple

import pandas as pd
import scipy.sparse as sp
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Sparse2Corpus, corpus2csc
from penelope import corpus as pc
from penelope.corpus.dtm.convert import id2token2token2id
from penelope.utility.utils import csr2bow

# pylint: disable=unused-argument

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
    vocabulary: Dictionary = Dictionary()
    if id2token is not None:
        vocabulary.token2id = id2token2token2id(id2token)
    vocabulary.add_documents(tokens for _, tokens in source)
    return vocabulary


class Id2TokenMissingError(NotImplementedError):
    ...


class TranslateCorpus:
    def translate(
        self,
        source: Any,
        *,
        id2token: Mapping[int, str] = None,
        document_index: pd.DataFrame,
        **vectorize_opts,
    ) -> Tuple[Sparse2Corpus, Dictionary]:

        """Gensim doc says:
        "corpus : iterable of list of (int, float), optional
        Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        If you have a CSC in-memory matrix, you can convert it to a
        streamed corpus with the help of gensim.matutils.Sparse2Corpus.
        If not given, the model is left untrained ...."
        """
        vocabulary: Dictionary = None
        corpus: Sparse2Corpus = None

        if isinstance(source, Sparse2Corpus):
            corpus = source
        elif isinstance(source, pc.VectorizedCorpus):
            corpus = Sparse2Corpus(source.data, documents_columns=False)
        elif sp.issparse(source):
            corpus = Sparse2Corpus(source, documents_columns=False)
        else:
            """Assumes stream of (document, tokens)"""
            vocabulary = from_stream_of_tokens_to_dictionary(source, id2token)
            corpus = from_stream_of_tokens_to_sparse2corpus(source, vocabulary)

        if vocabulary is None:
            """Build from corpus, `id2token` must be supplied"""
            if id2token is None:
                raise Id2TokenMissingError()
            vocabulary: Dictionary = Dictionary.from_corpus(csr2bow(corpus.sparse), id2word=id2token)

        return corpus, vocabulary
