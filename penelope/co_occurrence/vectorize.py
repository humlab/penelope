from collections import Counter
from typing import Iterator, Mapping, Tuple

import numpy as np
import scipy
from penelope.corpus import Token2Id
from sklearn.feature_extraction.text import CountVectorizer


class WindowsCoOccurrenceVectorizer:
    """Creates a term-term-matrix from a sequence of windows (tokens)"""

    # FIXME Add WordWindowsCounter ()
    def __init__(self, vocabulary: Token2Id):

        self.window_counts_global: Counter = Counter()
        # self.window_counts_document: Mapping[int, int] = None
        self.vectorizer: CountVectorizer = CountVectorizer(
            tokenizer=lambda x: x,
            vocabulary=vocabulary.data,
            lowercase=False,
            dtype=np.uint16,
        )

    def fit_transform(self, windows: Iterator[Iterator[str]]) -> Tuple[scipy.sparse.spmatrix, Mapping[int, int]]:
        """Fits windows generated from a __single__ document"""

        window_term_matrix: scipy.sparse.spmatrix = self.vectorizer.fit_transform(windows)

        term_term_matrix: scipy.sparse.spmatrix = scipy.sparse.triu(
            np.dot(window_term_matrix.T, window_term_matrix),
            1,
        )

        window_counts_document: Mapping[int, int] = self._get_window_counts(window_term_matrix)

        self.window_counts_global.update(window_counts_document)

        return term_term_matrix, window_counts_document

    def _get_window_counts(self, window_term_matrix: scipy.sparse.spmatrix) -> Mapping[int, int]:
        """Returns window counts for non-zero tokens in window_term_matrix"""
        win_counts = (window_term_matrix != 0).sum(axis=0).A1
        nz_indicies = win_counts.nonzero()[0]
        _token_windows_counts = {i: win_counts[i] for i in nz_indicies}
        return _token_windows_counts
