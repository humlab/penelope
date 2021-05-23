from collections import Counter
from typing import Any, Iterator, Mapping, Tuple

import numpy as np
import scipy
from penelope.corpus import Token2Id
from sklearn.feature_extraction.text import CountVectorizer


class WindowsCoOccurrenceVectorizer:
    """Creates a term-term-matrix from a sequence of windows (tokens)"""

    def __init__(self, vocabulary: Token2Id, dtype: Any = np.uint32):

        self.corpus_token_window_counts: Counter = Counter()
        self.vocabulary: Token2Id = vocabulary
        self.dtype = dtype

    def fit_transform(self, windows: Iterator[Iterator[str]]) -> Tuple[scipy.sparse.spmatrix, Mapping[int, int]]:
        """Fits windows generated from a __single__ document"""

        # self.vocabulary.ingest(itertools.chain(*windows))

        vectorizer: CountVectorizer = CountVectorizer(
            tokenizer=lambda x: x,
            vocabulary=self.vocabulary.data,
            lowercase=False,
            dtype=self.dtype,
        )
        window_term_matrix: scipy.sparse.spmatrix = vectorizer.fit_transform(windows)

        term_term_matrix: scipy.sparse.spmatrix = scipy.sparse.triu(
            np.dot(window_term_matrix.T, window_term_matrix),
            1,
        )

        document_token_window_count_matrix: Mapping[int, int] = self._get_window_counts(window_term_matrix)

        self.corpus_token_window_counts.update(document_token_window_count_matrix)

        return term_term_matrix, document_token_window_count_matrix

    def _get_window_counts(self, window_term_matrix: scipy.sparse.spmatrix) -> Mapping[int, int]:
        """Returns window counts for non-zero tokens in window_term_matrix"""
        win_counts = (window_term_matrix != 0).sum(axis=0).A1
        nz_indicies = win_counts.nonzero()[0]
        _token_windows_counts = {i: win_counts[i] for i in nz_indicies}
        return _token_windows_counts
