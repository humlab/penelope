from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple, Union

import numpy as np
import textacy

from .interface import IVectorizedCorpus, IVectorizedCorpusProtocol


class ISlicedCorpusProtocol(IVectorizedCorpusProtocol):
    def slice_by_tf(self, n_count: int) -> IVectorizedCorpus:
        ...

    def slice_by_n_top(self, n_top: int) -> IVectorizedCorpus:
        ...

    def slice_by_document_frequency(self, max_df=1.0, min_df=1, max_n_terms=None) -> IVectorizedCorpus:
        ...

    def slice_by(self, px) -> IVectorizedCorpus:
        ...

    def slice_by_indices(self, indices: Sequence[int], inplace=False) -> IVectorizedCorpus:
        ...

    def slice_by_term_frequency(self, threshold: Union[int, float], inplace=False) -> IVectorizedCorpus:
        ...

    @staticmethod
    def where_is_above_threshold_with_keeps(
        values: np.ndarray, threshold: Union[int, float], keep_indices: List[int] = None
    ) -> np.ndarray:
        ...

    def term_frequencies_greater_than_or_equal_to_threshold(self, threshold: Union[int, float]) -> np.ndarray:
        ...

    def compress(
        self, tf_threshold: int = 1, extra_keep_ids: List[int] = None, inplace=False
    ) -> Tuple[IVectorizedCorpus, Mapping[int, int], Sequence[int]]:
        ...


class SliceMixIn:

    # @autojit
    def slice_by_tf(self: ISlicedCorpusProtocol, n_count: int) -> IVectorizedCorpus:
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

        indices: np.ndarray = np.argwhere(self.term_frequency >= n_count).ravel()
        return self.slice_by_indices(indices)

    def slice_by_n_top(self: ISlicedCorpusProtocol, n_top: int) -> IVectorizedCorpus:
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
        return self.slice_by_indices(self.nlargest(n_top=n_top))

    def slice_by_document_frequency(
        self: ISlicedCorpusProtocol, max_df=1.0, min_df=1, max_n_terms=None
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

        bag_term_matrix, token2id = textacy.vsm.matrix_utils.filter_terms_by_df(
            self.bag_term_matrix, self.token2id, max_df=max_df, min_df=min_df, max_n_terms=max_n_terms
        )
        overridden_term_frequency = (
            self._overridden_term_frequency[list(sorted(self.token2id[w] for w in token2id))]
            if self._overridden_term_frequency is not None
            else None
        )

        corpus = self.create(bag_term_matrix, token2id, self.document_index, overridden_term_frequency)

        return corpus

    # @autojit
    def slice_by(self: ISlicedCorpusProtocol, px) -> IVectorizedCorpus:
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

        corpus = self.slice_by_indices(indices)

        return corpus

    # @autojit
    def slice_by_indices(self: ISlicedCorpusProtocol, indices: Sequence[int], inplace=False) -> IVectorizedCorpus:
        """Create (or modifies inplace) a subset corpus from given `indices`"""

        if indices is None:
            indices = []

        if len(indices) == self.bag_term_matrix.shape[1]:
            return self

        indices.sort()

        bag_term_matrix = self.bag_term_matrix[:, indices]
        token2id = {self.id2token[indices[i]]: i for i in range(0, len(indices))}

        overridden_term_frequency = (
            self._overridden_term_frequency[indices] if self._overridden_term_frequency is not None else None
        )

        if not inplace:
            corpus = self.create(bag_term_matrix, token2id, self.document_index, overridden_term_frequency)
            return corpus

        self._bag_term_matrix = bag_term_matrix
        self._token2id = token2id
        self._id2token = None
        self._overridden_term_frequency = overridden_term_frequency

        return self

    def slice_by_term_frequency(
        self: ISlicedCorpusProtocol, threshold: Union[int, float], inplace=False
    ) -> IVectorizedCorpus:
        """Returns subset of corpus where low frequenct words are filtered out"""
        indices = self.term_frequencies_greater_than_or_equal_to_threshold(threshold)
        corpus: IVectorizedCorpus = self.slice_by_indices(indices, inplace=inplace)
        return corpus

    @staticmethod
    def where_is_above_threshold_with_keeps(
        values: np.ndarray, threshold: Union[int, float], keep_indices: List[int] = None
    ) -> np.ndarray:
        """Returns indices for values above threshold or in keeps"""
        mask = values >= threshold
        if keep_indices:
            mask[keep_indices] = True
        indices = np.argwhere(mask).ravel()
        return indices

    def term_frequencies_greater_than_or_equal_to_threshold(
        self: ISlicedCorpusProtocol, threshold: Union[int, float], keep_indices: List[int] = None
    ) -> np.ndarray:
        """Returns indices of words having a frequency below a given threshold"""
        indices = self.where_is_above_threshold_with_keeps(
            self.term_frequency, threshold, keep_indices=keep_indices
        ).ravel()
        return indices

    def compress(
        self: ISlicedCorpusProtocol,
        tf_threshold: int = 1,
        extra_keep_ids: List[int] = None,
        inplace=False,
    ) -> Tuple[IVectorizedCorpus, Mapping[int, int], Sequence[int]]:
        """Compresses corpus by eliminating zero-TF terms.

        Returns:
            Tuple[IVectorizedCorpus, Mapping[int,int], Sequence[int]]: compressed corpus, mapping between old/new vocabularies and affected original indices
        """
        keep_ids = self.term_frequencies_greater_than_or_equal_to_threshold(tf_threshold, keep_indices=extra_keep_ids)

        if len(keep_ids) == 0:
            return self, {}, []

        ids_translation: Mapping[int, int] = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}

        corpus: IVectorizedCorpus = self.slice_by_indices(keep_ids, inplace=inplace)

        return (corpus, ids_translation, keep_ids)
