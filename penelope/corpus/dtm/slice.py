from __future__ import annotations

from heapq import nlargest
from typing import Sequence, Union

import numpy as np
import textacy

from .interface import IVectorizedCorpus, IVectorizedCorpusProtocol


class SliceMixIn:

    # @autojit
    def slice_by_n_count(self: IVectorizedCorpusProtocol, n_count: int) -> IVectorizedCorpus:
        """Create a subset corpus where words having a count less than 'n_count' are removed

        Parameters
        ----------
        n_count : int
            Specifies min word count to keep.

        Returns
        -------
        VectorizedCorpus
            Subset of self where words having a count less than 'n_count' are removed
        """

        tokens = set(w for w, c in self.term_frequency_mapping.items() if c >= n_count)

        def _px(w):
            return w in tokens

        return self.slice_by(_px)

    def slice_by_n_top(self: IVectorizedCorpusProtocol, n_top) -> IVectorizedCorpus:
        """Create a subset corpus that only contains most frequent `n_top` words

        Parameters
        ----------
        n_top : int
            Specifies specifies number of top words to keep.

        Returns
        -------
        VectorizedCorpus
            Subset of self where words having a count less than 'n_count' are removed
        """
        tokens = set(nlargest(n_top, self.term_frequency_mapping, key=self.term_frequency_mapping.get))

        def _px(w):
            return w in tokens

        return self.slice_by(_px)

    # def doc_freqs(self):
    #     """ Count number of occurrences of each value in array of non-negative ints. """
    #     return np.bincount(self.doc_term_matrix.indices, minlength=self.n_terms)

    # def slice_by_document_frequency(self, min_df, max_df):
    #     min_doc_count = min_df if isinstance(min_df, int) else int(min_df * self.n_docs)
    #     dfs = self.doc_freqs()
    #     mask = np.ones(self.n_terms, dtype=bool)
    #     if min_doc_count > 1:
    #         mask &= dfs >= min_doc_count
    #     # map old term indices to new ones
    #     new_indices = np.cumsum(mask) - 1
    #     token2id = {
    #         term: new_indices[old_index]
    #         for term, old_index in self.token2id.items()
    #             if mask[old_index]
    #     }
    #     kept_indices = np.where(mask)[0]
    #     return (self.bag_term_matrix[:, kept_indices], token2id)

    def slice_by_document_frequency(
        self: IVectorizedCorpusProtocol, max_df=1.0, min_df=1, max_n_terms=None
    ) -> IVectorizedCorpus:
        """Creates a subset corpus where common/rare terms are filtered out.

        Textacy util function filter_terms_by_df is used for the filtering.

        See https://chartbeat-labs.github.io/textacy/build/html/api_reference/vsm_and_tm.html.

        Parameters
        ----------
        max_df : float, optional
            Max number of docs or fraction of total number of docs, by default 1.0
        min_df : int, optional
            Max number of docs or fraction of total number of docs, by default 1
        max_n_terms : in optional
            [description], by default None
        """
        sliced_bag_term_matrix, token2id = textacy.vsm.matrix_utils.filter_terms_by_df(
            self.bag_term_matrix, self.token2id, max_df=max_df, min_df=min_df, max_n_terms=max_n_terms
        )
        term_frequency_mapping = {w: c for w, c in self.term_frequency_mapping.items() if w in token2id}

        v_corpus = self.create(sliced_bag_term_matrix, token2id, self.document_index, term_frequency_mapping)

        return v_corpus

    # @autojit
    def slice_by(self: IVectorizedCorpusProtocol, px) -> IVectorizedCorpus:
        """Create a subset corpus based on predicate `px`

        Parameters
        ----------
        px : str -> bool
            Predicate that tests if a word should be kept.

        Returns
        -------
        VectorizedCorpus
            Subset containing words for which `px` evaluates to true.
        """
        indices = [self.token2id[w] for w in self.token2id.keys() if px(w)]

        v_corpus = self.slice_by_indicies(indices)

        return v_corpus

    # @autojit
    def slice_by_indicies(self: IVectorizedCorpusProtocol, indices: Sequence[int]) -> IVectorizedCorpus:
        """Create a subset corpus from given `indices`"""

        indices.sort()

        sliced_bag_term_matrix = self.bag_term_matrix[:, indices]
        token2id = {self.id2token[indices[i]]: i for i in range(0, len(indices))}
        term_frequency_mapping = {w: c for w, c in self.term_frequency_mapping.items() if w in token2id}

        v_corpus = self.create(sliced_bag_term_matrix, token2id, self.document_index, term_frequency_mapping)

        return v_corpus

    def slice_by_term_frequency(self, threshold: Union[int, float]) -> IVectorizedCorpus:
        """Returns subset of corpus where low frequenct words are fioltered out"""
        indicies = self.term_frequencies_below_threshold(threshold)
        corpus: IVectorizedCorpus = self.slice_by_indicies(indicies)
        return corpus

    def term_frequencies_below_threshold(self, threshold: Union[int, float]) -> np.ndarray:
        """Returns indicies of words having a frequency below a given threshold"""
        indicies = np.argwhere(self.term_frequencies >= threshold).ravel()
        return indicies
