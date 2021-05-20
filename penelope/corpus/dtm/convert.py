from __future__ import annotations

import pandas as pd

from .interface import IVectorizedCorpusProtocol
from ..token2id import Token2Id


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
