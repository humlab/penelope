from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple

import numpy as np
import scipy.sparse as sp

try:
    from gensim.corpora.dictionary import Dictionary
    from gensim.matutils import Sparse2Corpus, corpus2csc
except (ImportError, NameError):

    class Dictionary(dict):
        @staticmethod
        def from_corpus(corpus, id2word=None):  # pylint: disable=unused-argument
            raise ModuleNotFoundError()

    class Sparse2Corpus:
        # Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
        # Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
        # Modified code;
        def __init__(self, sparse, documents_columns=True):
            self.sparse = sparse.tocsc() if documents_columns else sparse.tocsr().T

        def __iter__(self):
            for indprev, indnow in zip(self.sparse.indptr, self.sparse.indptr[1:]):
                yield list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

        def __len__(self):
            return self.sparse.shape[1]

        def __getitem__(self, document_index):
            indprev = self.sparse.indptr[document_index]
            indnow = self.sparse.indptr[document_index + 1]
            return list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

    def corpus2csc(corpus, num_terms=None, dtype=np.float64, num_docs=None, num_nnz=None, printprogress=0):
        # Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
        # Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
        # (code removed)
        return None


def _id2token2token2id(id2token: Mapping[int, str]) -> dict:
    if id2token is None:
        return None
    if hasattr(id2token, 'token2id'):
        return id2token.token2id
    token2id: dict = {v: k for k, v in id2token.items()}
    return token2id


GensimBowCorpus = Iterable[Iterable[Tuple[int, float]]]


def from_stream_of_tokens_to_sparse2corpus(source: Any, vocabulary: Dictionary | dict) -> Sparse2Corpus:

    if not hasattr(vocabulary, 'doc2bow'):
        vocabulary: Dictionary = _from_token2id_to_dictionary(vocabulary)

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


def _from_token2id_to_dictionary(token2id: Mapping[str, int]) -> Dictionary:

    if isinstance(token2id, Dictionary):
        return token2id

    dictionary: Dictionary = Dictionary()
    dictionary.token2id = token2id

    return dictionary


def from_id2token_to_dictionary(id2token: dict) -> Dictionary:
    """Creates a `Dictionary` from a id2token dict."""
    return _from_token2id_to_dictionary(_id2token2token2id(id2token))
