from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple

import pandas as pd
import scipy.sparse as sp
from more_itertools import peekable

from penelope.vendor.gensim_api import corpora as gensim_corpora
from penelope.vendor.textacy_api import Vectorizer

from ..token2id import id2token2token2id
from ..tokenized_corpus import TokenizedCorpus
from .corpus import VectorizedCorpus
from .vectorizer import CorpusVectorizer, DocumentTermsStream, VectorizeOpts

"""Ways to vectorize:
1. penelope.corpus.CorpusVectorizer
    USES sklearn.feature_extraction.text.CountVectorizer

2. engine_gensim.convert.TranslateCorpus -> Sparse2Corpus, Dictionary
    Dictionary.doc2bow, corpus2csc

2. textacy.Vectorizer -> sp.csr_matrix, id_to_term
    Has lots of options! Easy to translate to VectorizedCorpus

Returns:
    [type]: [description]
"""


def from_sparse2corpus(
    source: gensim_corpora.Sparse2Corpus, *, token2id: Mapping[str, int], document_index: pd.DataFrame
) -> VectorizedCorpus:
    corpus: VectorizedCorpus = VectorizedCorpus(
        bag_term_matrix=source.sparse.tocsr().T, token2id=token2id, document_index=document_index
    )
    return corpus


def to_sparse2corpus(corpus: VectorizedCorpus):

    return gensim_corpora.Sparse2Corpus(corpus.data, documents_columns=False)


def from_spmatrix(
    source: sp.spmatrix, *, token2id: Mapping[str, int], document_index: pd.DataFrame
) -> VectorizedCorpus:
    corpus = VectorizedCorpus(bag_term_matrix=source, token2id=token2id, document_index=document_index)
    return corpus


def from_tokenized_corpus(
    source: TokenizedCorpus, *, document_index: pd.DataFrame, vectorize_opts: VectorizeOpts
) -> VectorizedCorpus:
    corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(
        source, vocabulary=source.token2id, document_index=document_index, **vectorize_opts.props
    )
    return corpus


def from_stream_of_tokens(
    source: Iterable[Iterable[str]],
    *,
    token2id: Mapping[str, int],
    document_index: pd.DataFrame,
    vectorize_opts: VectorizeOpts,
) -> VectorizedCorpus:

    vectorizer: Vectorizer = Vectorizer(
        min_df=vectorize_opts.min_df,
        max_df=vectorize_opts.max_df,
        max_n_terms=vectorize_opts.max_tokens,
        vocabulary_terms=token2id,
        # tf_type: Literal["linear", "sqrt", "log", "binary"] = "linear",
        # idf_type: Optional[Literal["standard", "smooth", "bm25"]] = None,
        # dl_type: Optional[Literal["linear", "sqrt", "log"]] = None,
        # norm: Optional[Literal["l1", "l2"]] = None,
    )

    bag_term_matrix: sp.spmatrix = vectorizer.fit_transform(source)

    if token2id is None:
        token2id = id2token2token2id(vectorizer.id_to_term)

    corpus: VectorizedCorpus = VectorizedCorpus(
        bag_term_matrix=bag_term_matrix, token2id=token2id, document_index=document_index
    )

    return corpus


def from_stream_of_filename_tokens(
    source: DocumentTermsStream,
    *,
    token2id: Mapping[str, int],
    document_index: pd.DataFrame,
    vectorize_opts: VectorizeOpts,
) -> VectorizedCorpus:
    assert vectorize_opts.already_tokenized
    corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(
        source, vocabulary=token2id, document_index=document_index, **vectorize_opts.props
    )

    return corpus


def from_stream_of_text(
    source: Iterable[str], *, token2id: Mapping[str, int], document_index: pd.DataFrame, vectorize_opts: VectorizeOpts
) -> VectorizedCorpus:
    assert not vectorize_opts.already_tokenized
    corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(
        source, vocabulary=token2id, document_index=document_index, **vectorize_opts.props
    )

    return corpus


# pylint: disable=too-many-return-statements


class TranslateCorpus:
    @staticmethod
    def translate(
        source: Any,
        *,
        token2id: Mapping[str, int] = None,
        document_index: pd.DataFrame,
        vectorize_opts: VectorizeOpts,
    ) -> VectorizedCorpus:
        """Translate source into a `VectorizedCorpus``"""

        if isinstance(source, VectorizedCorpus):
            return source

        if isinstance(source, sp.spmatrix):
            return from_spmatrix(source, token2id=token2id, document_index=document_index)

        # if type(source).__name__.endswith('Sparse2Corpus'):
        if isinstance(source, gensim_corpora.Sparse2Corpus):
            return from_sparse2corpus(source, token2id=token2id, document_index=document_index)

        if isinstance(source, TokenizedCorpus):
            return from_tokenized_corpus(source, document_index=document_index, vectorize_opts=vectorize_opts)

        source: peekable = peekable(source)
        head: Any = source.peek()

        if isinstance(head, Tuple):
            """Vectorize using CorpusVectorizer, stream must be Iterable[Tuple[document-name, Iterable[str]]]"""
            return from_stream_of_filename_tokens(
                source, token2id=token2id, document_index=document_index, vectorize_opts=vectorize_opts
            )

        if isinstance(head, str):
            """Vectorize using CorpusVectorizer, stream must be Iterable[Tuple[document-name, Iterable[str]]]"""
            return from_stream_of_text(
                source, token2id=token2id, document_index=document_index, vectorize_opts=vectorize_opts
            )

        return from_stream_of_tokens(
            source, token2id=token2id, document_index=document_index, vectorize_opts=vectorize_opts
        )
