from collections import Counter
from typing import Iterator, Mapping

import numpy as np
import scipy
from penelope.corpus import Token2Id
from sklearn.feature_extraction.text import CountVectorizer


class WindowsCoOccurrenceVectorizer:
    """Creates a term-term-matrix from a sequence of windows (tokens)"""

    # FIXME Add WordWindowsCounter ()
    def __init__(self, vocabulary: Token2Id):

        self.window_counts_global: Counter = Counter()
        self.vectorizer: CountVectorizer = CountVectorizer(
            tokenizer=lambda x: x, vocabulary=vocabulary.data, lowercase=False, dtype=np.uint16
        )

    def fit_transform(self, windows: Iterator[Iterator[str]]) -> scipy.sparse.spmatrix:
        """Fits windows generated from a __single__ document"""

        window_term_matrix = self.vectorizer.fit_transform(windows)

        term_term_matrix = scipy.sparse.triu(
            np.dot(window_term_matrix.T, window_term_matrix),
            1,
        )

        window_counts = self._get_window_counts(window_term_matrix)

        self.window_counts_global.update(window_counts)

        return term_term_matrix, window_counts

    def _get_window_counts(self, bag_term_matrix) -> Mapping[int, int]:
        """Returns window counts for non-zero tokens in bag_term_matrix"""
        win_counts = (bag_term_matrix != 0).sum(axis=0).A1
        nz_indicies = win_counts.nonzero()[0]
        _token_windows_counts = {i: win_counts[i] for i in nz_indicies}
        return _token_windows_counts
