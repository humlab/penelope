from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Tuple

import numpy as np
import pandas as pd
import scipy
from penelope.common.keyness import KeynessMetric, partitioned_significances
from penelope.type_alias import CoOccurrenceDataFrame, DocumentIndex
from penelope.utility.utils import create_instance

from ..token2id import Token2Id
from .interface import IVectorizedCorpusProtocol

if TYPE_CHECKING:
    from .vectorized_corpus import VectorizedCorpus


def empty_data() -> pd.DataFrame:

    frame: pd.DataFrame = pd.DataFrame(
        data={
            'document_id': pd.Series(data=[], dtype=np.int32),
            'token_id': pd.Series(data=[], dtype=np.int32),
            'value': pd.Series(data=[], dtype=np.int32),
            'time_period': pd.Series(data=[], dtype=np.int32),
            'w1_id': pd.Series(data=[], dtype=np.int32),
            'w2_id': pd.Series(data=[], dtype=np.int32),
        }
    )
    return frame


class CoOccurrenceMixIn:
    @staticmethod
    def create_co_occurrence_vocabulary(co_occurrences: pd.DataFrame, token2id: Token2Id) -> Tuple[dict, pd.Series]:
        """Returns a new vocabulary for word-pairs in `co_occurrences`"""

        to_token = token2id.id2token.get
        token_pairs: pd.DataFrame = co_occurrences[["w1_id", "w2_id"]].drop_duplicates().reset_index(drop=True)
        token_pairs["token_id"] = token_pairs.index
        token_pairs["token"] = token_pairs.w1_id.apply(to_token) + "/" + token_pairs.w2_id.apply(to_token)

        """Create a new vocabulary"""
        vocabulary = token_pairs.set_index("token").token_id.to_dict()

        vocabulary_mapping: Mapping[Tuple[int, int], int] = token_pairs.set_index(['w1_id', 'w2_id']).token_id.to_dict()

        return vocabulary, vocabulary_mapping

    def to_co_occurrences(
        self: IVectorizedCorpusProtocol, source_token2id: Token2Id, partition_key: str = None
    ) -> pd.DataFrame:
        """Creates a co-occurrence data frame from a vectorized self (DTM)

        NOTE:
            source_token2id [Token2Id]: Vocabulary for source corpus
            self.token2id [dict]:       Vocabulary of co-occuring token pairs
        """

        partition_key = partition_key or ('time_period' if 'time_period' in self.document_index.columns else 'year')

        if 0 in self.data.shape:
            return empty_data()

        coo = self.data.tocoo(copy=False)
        df = pd.DataFrame(
            {
                # 'document_id': coo.row,
                'document_id': coo.row.astype(np.int32),
                'token_id': coo.col.astype(np.int32),
                'value': coo.data,
            }
        )

        """Add a time period column that can be used as a pivot column"""
        df['time_period'] = self.document_index.loc[df.document_id][partition_key].astype(np.int16).values

        """Decode w1/w2 token pair"""
        fg = self.id2token.get

        df['token'] = df.token_id.apply(fg)

        ws = pd.DataFrame(df.token.str.split('/', 1).tolist(), columns=['w1', 'w2'], index=df.index)
        df = df.assign(w1=ws.w1, w2=ws.w2)

        # """Decode w1 and w2 tokens using supplied token2id"""
        sg = source_token2id.get
        df['w1_id'] = df.w1.apply(sg).astype(np.int32)
        df['w2_id'] = df.w2.apply(sg).astype(np.int32)

        df.drop(columns=["token", "w1", "w2"], inplace=True)
        # df = df.reset_index()

        return df

    @staticmethod
    def from_co_occurrences(
        *, co_occurrences: CoOccurrenceDataFrame, document_index: DocumentIndex, token2id: Token2Id
    ) -> VectorizedCorpus:
        """Creates a co-occurrence DTM corpus from a co-occurrences data frame."""
        if not isinstance(token2id, Token2Id):
            token2id = Token2Id(data=token2id)

        vocabulary, vocabulay_mapping = CoOccurrenceMixIn.create_co_occurrence_vocabulary(co_occurrences, token2id)

        """Set document_id as unique key for DTM document index """
        document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

        """Make certain that the matrix gets right shape (to avoid offset errors)"""
        shape = (len(document_index), len(vocabulary))

        fg = vocabulay_mapping.get
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
        cls: type = create_instance("penelope.corpus.dtm.vectorized_corpus.VectorizedCorpus")
        corpus = cls(matrix, token2id=vocabulary, document_index=document_index)
        return corpus

    def to_keyness_co_occurrences(
        self: IVectorizedCorpusProtocol, keyness: KeynessMetric, token2id: Token2Id, pivot_key: str
    ) -> pd.DataFrame:
        """Returns co-occurrence data frame with PPMI values.

        Keyness values are computed for each partition as specified by pivot_key.

        Note: Corpus must be a co-occurrences corpus!
              Tokens must be of the form "w1/w2".
              Supplied token2id must be vocabulary for single words "w1", "w2", ...

        Args:
            token2id (Token2Id): [description]
            pivot_key (str): [description]

        Returns:
            pd.DataFrame: [description]
        """

        co_occurrences: pd.DataFrame = partitioned_significances(
            self.to_co_occurrences(token2id),
            keyness_metric=keyness,
            pivot_key=pivot_key,
            vocabulary_size=len(token2id),
        )
        return co_occurrences

    def to_keyness_co_occurrence_corpus(
        self: IVectorizedCorpusProtocol, keyness: KeynessMetric, token2id: Token2Id, pivot_key: str
    ) -> VectorizedCorpus:
        """Returns

        Args:
            token2id (Token2Id): [description]
            pivot_key (str): [description]
            shape (Tuple[int, int]): [description]

        Returns:
            pd.DataFrame: [description]
        """
        co_occurrences: pd.DataFrame = partitioned_significances(
            self.to_co_occurrences(token2id),
            pivot_key=pivot_key,
            keyness_metric=keyness,
            vocabulary_size=len(token2id),
        )
        matrix = scipy.sparse.coo_matrix(
            (
                co_occurrences.value,
                (
                    co_occurrences.document_id.astype(np.int32),
                    co_occurrences.token_id.astype(np.int32),
                ),
            ),
            shape=(len(self.document_index), len(self.token2id)),
        )

        cls: type = create_instance("penelope.corpus.dtm.vectorized_corpus.VectorizedCorpus")
        corpus = cls(matrix, token2id=self.token2id, document_index=self.document_index)
        return corpus
