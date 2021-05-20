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

        self.global_token_windows_counts: Counter = Counter()
        self.last_token_windows_counts: Mapping[int, int] = None
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

        window_counts = self._get_token_window_counts(window_term_matrix)

        self.global_token_windows_counts.update(window_counts)

        return term_term_matrix, window_counts

    def _get_token_window_counts(self, bag_term_matrix) -> Mapping[int, int]:
        """Returns window counts for non-zero tokens in bag_term_matrix"""
        win_counts = (bag_term_matrix != 0).sum(axis=0).A1
        nz_indicies = win_counts.nonzero()[0]
        _token_windows_counts = {i: win_counts[i] for i in nz_indicies}
        return _token_windows_counts


# def term_term_matrices_to_co_occurrences_corpus(
#     document_id: int,
#     term_term_matrix: scipy.sparse.spmatrix,
#     token2id: Token2Id,
# ):
#     ...
#     """Convert a sequence of TTM to a CC-Corpus"""
#     token2id.ingest()
#     w1_id = term_term_matrix.row
#     w2_id = term_term_matrix.col
#     values = term_term_matrix.data

#     shape = (len(document_index), len(vocabulary))
#     matrix = scipy.sparse.coo_matrix(
#         (
#             co_occurrences.value.astype(np.uint16),
#             (
#                 co_occurrences.document_id.astype(np.uint32),
#                 token_ids.astype(np.uint32),
#             ),
#         ),
#         shape=shape,
#     )
