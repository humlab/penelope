from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Tuple

import numpy as np
import pandas as pd
import scipy

from penelope.common.keyness import KeynessMetric, partitioned_significances
from penelope.corpus.dtm.interface import IVectorizedCorpusProtocol
from penelope.utility import create_class, deprecated

from ..token2id import Token2Id
from .ttm import CoOccurrenceVocabularyHelper

if TYPE_CHECKING:
    from .corpus import VectorizedCorpus


WORD_PAIR_DELIMITER = "/"

# pylint: disable=no-member


class LegacyCoOccurrenceMixIn:
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

        corpus = self.corpus_class(bag_term_matrix=matrix, token2id=token2id, document_index=self.document_index)

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
        corpus_class: type = create_class("penelope.corpus.dtm.corpus.VectorizedCorpus")
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

    @deprecated
    @staticmethod
    def from_co_occurrences(
        *, co_occurrences: pd.DataFrame, document_index: pd.DataFrame, token2id: Token2Id
    ) -> Tuple[VectorizedCorpus, Mapping[Tuple[int, int], int]]:
        """Creates a co-occurrence DTM corpus from a co-occurrences data frame.

           A "word-pair token" in the corpus' vocabulary has the form "w1 WORD_PAIR_DELIMITER w2".

           The mapping between the two vocabulary is stored in self.payload['vocabs_mapping]
           The mapping translates identities for (w1,w2) to identity for "w1 WORD_PAIR_DELIMITER w2".


        Args:
            co_occurrences (CoOccurrenceDataFrame): [description]
            document_index (DocumentIndex): [description]
            token2id (Token2Id): source corpus vocabulary

        Returns:
            VectorizedCorpus: The co-occurrence corpus
        """

        if not isinstance(token2id, Token2Id):
            token2id = Token2Id(data=token2id)

        vocabulary, vocabs_mapping = CoOccurrenceVocabularyHelper.create_pair2id(co_occurrences, token2id)

        """Set document_id as unique key for DTM document index """
        document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

        """Make certain that the matrix gets right shape (to avoid offset errors)"""

        shape = (len(document_index), len(vocabulary))

        if len(vocabulary) == 0:
            matrix = scipy.sparse.coo_matrix(([], (co_occurrences.document_id, [])), shape=shape)
        else:
            fg = vocabs_mapping.get
            matrix = scipy.sparse.coo_matrix(
                (
                    co_occurrences.value.astype(np.int32),
                    (
                        co_occurrences.document_id.astype(np.int32),
                        co_occurrences[['w1_id', 'w2_id']].apply(lambda x: fg((x[0], x[1])), axis=1),
                    ),
                ),
                shape=shape,
            )

        """Create the final corpus (dynamically to avoid cyclic dependency)"""
        corpus_cls: type = create_class("penelope.corpus.dtm.corpus.VectorizedCorpus")
        corpus: VectorizedCorpus = corpus_cls(
            bag_term_matrix=matrix,
            token2id=vocabulary,
            document_index=document_index,
        )

        corpus.remember(vocabs_mapping=vocabs_mapping)

        return corpus
