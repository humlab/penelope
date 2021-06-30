from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Tuple

import numpy as np
import pandas as pd
import scipy
from penelope.common.keyness import KeynessMetric, compute_hal_cwr_score, metrics, partitioned_significances
from penelope.corpus.dtm.interface import IVectorizedCorpusProtocol
from penelope.utility import create_instance, deprecated

from ..token2id import Token2Id
from .ttm import CoOccurrenceVocabularyHelper, empty_data

if TYPE_CHECKING:
    from .corpus import VectorizedCorpus


WORD_PAIR_DELIMITER = "/"


class LegacyCoOccurrenceKeynessMixIn:
    @deprecated
    def to_keyness_co_occurrences(
        self: IVectorizedCorpusProtocol,
        keyness: KeynessMetric,
        token2id: Token2Id,
        pivot_key: str,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Returns co-occurrence data frame with weighed values by significance metrics.

        Keyness values are computed for each partition as specified by pivot_key.

        Note: Corpus must be a co-occurrences corpus!
              Tokens must be of the form "w1 WORD_PAIR_DELIMITER w2".
              Supplied token2id must be vocabulary for single words "w1", "w2", ...

        Args:
            token2id (Token2Id): [description]
            pivot_key (str): [description]

        Returns:
            pd.DataFrame: [description]
        """
        co_occurrences: pd.DataFrame = self.to_co_occurrences(token2id)
        keyness_co_occurrences: pd.DataFrame = partitioned_significances(
            co_occurrences=co_occurrences,
            keyness_metric=keyness,
            pivot_key=pivot_key,
            document_index=self.document_index,
            vocabulary_size=len(token2id),
            normalize=normalize,
        )

        mg = self.get_token_ids_2_pair_id(token2id=token2id).get

        # co_occurrences['token_id'] = co_occurrences[['w1_id', 'w2_id']].apply(lambda x: mg((x[0], x[1])), axis=1)
        # faster:
        keyness_co_occurrences['token_id'] = [
            mg((x[0].item(), x[1].item())) for x in keyness_co_occurrences[['w1_id', 'w2_id']].to_records(index=False)
        ]

        return keyness_co_occurrences

    @deprecated
    def to_keyness_co_occurrence_corpus(
        self: IVectorizedCorpusProtocol,
        keyness: KeynessMetric,
        token2id: Token2Id,
        pivot_key: str,
        normalize: bool = False,
    ) -> VectorizedCorpus:
        """Returns a copy of the corpus where the values have been weighed by keyness metric.

        NOTE: Call only valid for co-occurrence corpus!

        Args:
            token2id (Token2Id): [description]
            pivot_key (str): [description]
            shape (Tuple[int, int]): [description]

        Returns:
            pd.DataFrame: [description]
        """

        co_occurrences: pd.DataFrame = self.to_keyness_co_occurrences(
            keyness=keyness,
            token2id=token2id,
            pivot_key=pivot_key,
            normalize=normalize,
        )

        """Map that translate pivot_key to document_id"""

        matrix = self._to_co_occurrence_matrix(co_occurrences, pivot_key)

        corpus = self.create_co_occurrence_corpus(matrix, token2id=token2id)

        return corpus

    @deprecated
    def _to_co_occurrence_matrix(self, co_occurrences: pd.DataFrame, pivot_key: str) -> scipy.sparse.spmatrix:

        """Map pivot_key value to document id (document index is already grouped)"""
        pg: Callable = {v: k for k, v in self.document_index[pivot_key].to_dict().items()}.get

        """Create a sparse matrix where rows are (pivoed) documets and columns are pair IDs"""
        matrix: scipy.sparse.spmatrix = scipy.sparse.coo_matrix(
            (
                co_occurrences.value,
                (
                    co_occurrences[pivot_key].apply(pg).astype(np.int32),
                    co_occurrences.token_id.astype(np.int32),
                ),
            ),
            shape=self.data.shape,
        )
        return matrix

    def create_co_occurrence_corpus(
        self, bag_term_matrix: scipy.sparse.spmatrix, token2id: Token2Id = None
    ) -> "VectorizedCorpus":
        corpus_class: type = create_instance("penelope.corpus.dtm.corpus.VectorizedCorpus")
        corpus: "VectorizedCorpus" = corpus_class(
            bag_term_matrix=bag_term_matrix,
            token2id=self.token2id,
            document_index=self.document_index,
        )

        vocabs_mapping: Any = self.payload.get("vocabs_mapping")

        if vocabs_mapping is None and token2id is not None:
            vocabs_mapping = self.get_token_ids_2_pair_id(token2id)

        if vocabs_mapping is not None:
            corpus.remember(vocabs_mapping=vocabs_mapping)

        return corpus
