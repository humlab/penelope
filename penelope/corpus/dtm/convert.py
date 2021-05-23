from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy
from penelope.type_alias import CoOccurrenceDataFrame, DocumentIndex
from penelope.utility.utils import create_instance

from ..token2id import Token2Id
from .interface import IVectorizedCorpusProtocol

if TYPE_CHECKING:
    from .vectorized_corpus import VectorizedCorpus


class CoOccurrenceMixIn:
    def to_co_occurrences(self: IVectorizedCorpusProtocol, source_token2id: Token2Id) -> pd.DataFrame:
        """Creates a co-occurrence data frame from a vectorized self (DTM)

        NOTE:
            source_token2id [Token2Id]: Vocabulary for source corpus
            self.token2id [dict]:       Vocabulary of co-occuring token pairs
        """

        coo = self.data.tocoo(copy=False)

        category_column = 'category' if 'category' in self.document_index.columns else 'document_id'

        df = (
            pd.DataFrame({category_column: coo.row, 'token_id': coo.col, 'value': coo.data})
            # .sort_values(['document_id', 'token_id'])
            # .reset_index(drop=True)
        )

        """Decode w1/w2 token pairs using own token2id"""
        fg = self.id2token.get

        df['token'] = df.token_id.apply(fg)

        ws = pd.DataFrame(df.token.str.split('/', 1).tolist(), columns=['w1', 'w2'], index=df.index)

        df = df.assign(w1=ws.w1, w2=ws.w2)

        """Decode w1 and w2 tokens using supplied token2id"""
        sg = source_token2id.get

        df['w1_id'] = df.w1.apply(sg)
        df['w2_id'] = df.w2.apply(sg)

        df = df[['document_id', 'w1_id', 'w2_id', 'value', 'token', 'w1', 'w2']]

        return df

    @staticmethod
    def from_co_occurrences(
        *,
        co_occurrences: CoOccurrenceDataFrame,
        document_index: DocumentIndex,
        token2id: Token2Id,
    ) -> VectorizedCorpus:
        """Creates a co-occurrence DTM corpus from a co-occurrence data frame."""
        if not isinstance(token2id, Token2Id):
            token2id = Token2Id(data=token2id)

        """Create distinct word-pair tokens and assign a token_id"""
        to_token = token2id.id2token.get
        token_pairs: pd.DataFrame = co_occurrences[["w1_id", "w2_id"]].drop_duplicates().reset_index(drop=True)
        token_pairs["token_id"] = token_pairs.index
        token_pairs["token"] = token_pairs.w1_id.apply(to_token) + "/" + token_pairs.w2_id.apply(to_token)

        """Create a new vocabulary"""
        vocabulary = token_pairs.set_index("token").token_id.to_dict()

        """Merge and assign token_id to co-occurring pairs"""
        token_ids: pd.Series = co_occurrences.merge(
            token_pairs.set_index(['w1_id', 'w2_id']),
            how='left',
            left_on=['w1_id', 'w2_id'],
            right_index=True,
        ).token_id

        """Set document_id as unique key for DTM document index """
        document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

        """Make certain that the matrix gets right shape (to avoid offset errors)"""
        shape = (len(document_index), len(vocabulary))
        matrix = scipy.sparse.coo_matrix(
            (
                co_occurrences.value.astype(np.uint16),
                (
                    co_occurrences.document_id.astype(np.uint32),
                    token_ids.astype(np.uint32),
                ),
            ),
            shape=shape,
        )

        """Create the final corpus (dynamically to avoid cyclic dependency)"""
        cls: type = create_instance("penelope.corpus.dtm.vectorized_corpus.VectorizedCorpus")
        corpus = cls(matrix, token2id=vocabulary, document_index=document_index)
        return corpus
