from collections import Counter
from typing import Iterator

import numpy as np
import scipy
from penelope.corpus import Token2Id
from sklearn.feature_extraction.text import CountVectorizer


class WindowsCoOccurrenceVectorizer:
    """Creates a term-term-matrix from a sequence of windows (tokens)"""

    # FIXME Add WordWindowsCounter ()
    def __init__(self, vocabulary: Token2Id):

        self.token_windows_counts: Counter = Counter()
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

        self._update_window_counts(window_term_matrix)

        return term_term_matrix

    def _update_window_counts(self, bag_term_matrix) -> None:
        """Updates tokens' window counts for non-zero tokens in bag_term_matrix"""
        win_counts = (bag_term_matrix != 0).sum(axis=0).A1
        nz_indicies = win_counts.nonzero()[0]
        self.token_windows_counts.update({i: win_counts[i] for i in nz_indicies})
