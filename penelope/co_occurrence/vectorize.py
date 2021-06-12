import array
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np
import scipy
from penelope.corpus import Token2Id


@dataclass
class WindowsCoOccurrenceOutput:

    term_term_matrix: scipy.sparse.spmatrix
    term_window_counter: Mapping[int, int]


class WindowsCoOccurrenceVectorizer:
    """Creates a term-term-matrix from a sequence of windows (tokens)"""

    def __init__(self, vocabulary: Token2Id, dtype: Any = np.int32):

        self.corpus_window_counts: Counter = Counter()
        self.vocabulary: Token2Id = vocabulary
        self.dtype = dtype

    def fit_transform(self, windows: Iterator[Iterator[str]]) -> WindowsCoOccurrenceOutput:
        """Fits windows generated from a __single__ document"""

        # self.vocabulary.ingest(itertools.chain(*windows))

        # vectorizer: CountVectorizer = CountVectorizer(
        #     tokenizer=lambda x: x,
        #     vocabulary=dict(self.vocabulary.data),
        #     lowercase=False,
        #     dtype=self.dtype,
        # )
        # window_term_matrix: scipy.sparse.spmatrix = vectorizer.fit_transform(windows)

        window_term_matrix: scipy.sparse.spmatrix = self.vectorize(windows)

        term_term_matrix: scipy.sparse.spmatrix = scipy.sparse.triu(
            np.dot(window_term_matrix.T, window_term_matrix),
            1,
        )

        term_window_counts: Mapping[int, int] = self.to_term_window_counter(window_term_matrix)

        self.corpus_window_counts.update(term_window_counts)

        return WindowsCoOccurrenceOutput(term_term_matrix, term_window_counts)

    def to_term_window_counter(self, window_term_matrix: scipy.sparse.spmatrix) -> Mapping[int, int]:
        """Returns tuples (token_id, window count) for non-zero tokens in window_term_matrix"""

        window_counts: np.ndarray = (window_term_matrix != 0).sum(axis=0).A1

        window_counter: Mapping[int, int] = {i: window_counts[i] for i in window_counts.nonzero()[0]}
        return window_counter

    def vectorize(self, windows: Iterator[Iterator[str]]) -> scipy.sparse.spmatrix:
        """Optimized/simplified version of sklearn.feature_extraction.text._count_vocab"""
        vocabulary = self.vocabulary
        indptr, jj = [], []

        values = array.array(str("i"))
        indptr.append(0)

        for window in windows:
            token_counter: Counter = Counter(vocabulary[t] for t in window)
            jj.extend(token_counter.keys())
            values.extend(token_counter.values())
            indptr.append(len(jj))

        jj = np.asarray(jj, dtype=np.int64)
        indptr = np.asarray(indptr, dtype=np.int32)
        values = np.frombuffer(values, dtype=np.intc)

        X = scipy.sparse.csr_matrix((values, jj, indptr), shape=(len(indptr) - 1, len(vocabulary)), dtype=self.dtype)
        X.sort_indices()
        return X
