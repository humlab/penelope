from typing import List, Union

import numpy as np
import pandas as pd
from penelope.corpus import Token2Id
from penelope.corpus.dtm.vectorized_corpus import VectorizedCorpus

IntOrStr = Union[int, str]


ID_COLUMNS: List[str] = ['token_id', 'w1_id', 'w2_id']
VALUE_COLUMNS: List[str] = ['value', 'n_tokens', 'n_raw_tokens']
TOKEN_COLUMNS: List[str] = ['w1', 'w2', 'token']


class CoOccurrenceHelper:
    def __init__(
        self,
        *,
        corpus: VectorizedCorpus,
        source_token2id: Token2Id,
        pivot_keys: Union[str, List[str]] = None,
    ):
        self.corpus: VectorizedCorpus = corpus
        self.source_token2id: Token2Id = source_token2id
        self.corpus_pivot_keys: List[str] = [pivot_keys] if isinstance(pivot_keys, str) else pivot_keys

        self.co_occurrences: pd.DataFrame = self.corpus.to_co_occurrences(source_token2id)  # .copy()
        self.data: pd.DataFrame = self.co_occurrences
        self.data_pivot_keys: List[str] = self.corpus_pivot_keys

    def reset(self) -> "CoOccurrenceHelper":
        self.data: pd.DataFrame = self.co_occurrences  # .copy()
        self.data_pivot_keys = self.corpus_pivot_keys
        return self

    def groupby(self, pivot_keys: Union[str, List[str]], normalize_key: str = 'n_raw_tokens') -> "CoOccurrenceHelper":
        """Groups co-occurrences data frame by given keys"""

        if self.data_pivot_keys:
            raise ValueError("Already grouped, please reset before calling again")

        if not pivot_keys:
            raise ValueError("pivot keys is not specified")

        data: pd.DataFrame = self.data

        if isinstance(pivot_keys, str):
            pivot_keys = [pivot_keys]

        pivot_keys = [g for g in pivot_keys if g in self.corpus.document_index.columns and g not in data.columns]

        if len(pivot_keys) == 0:
            raise ValueError("No keys to group by!")

        """Add grouping columns to data"""
        data: pd.DataFrame = data.merge(
            self.corpus.document_index[pivot_keys], left_on='document_id', right_index=True, how='inner'
        )

        """Group and sum up data"""
        data = data.groupby(pivot_keys + ID_COLUMNS)['value'].sum().reset_index()

        """Divide window counts with time-periods token counts"""
        data['value_n_t'] = data.value / pd.merge(
            data[pivot_keys],
            self.corpus.document_index.groupby(pivot_keys)[normalize_key].sum(),  # Yearly token counts
            left_on='year',
            right_index=True,
        )[normalize_key]

        self.data = data
        self.data_pivot_keys = pivot_keys

        return self

    def decode(self) -> "CoOccurrenceHelper":

        if 'w1' in self.data.columns:
            return self

        fg = self.source_token2id.id2token.get
        self.data["w1"] = self.data.w1_id.apply(fg)
        self.data["w2"] = self.data.w2_id.apply(fg)

        fg = self.corpus.id2token.get
        self.data["token"] = self.data.token_id.apply(fg)
        # faster:
        # self.data["token"] = self.data.w1 + '/' + self.data.w2

        return self

    def trunk_by_global_count(self, threshold: int) -> "CoOccurrenceHelper":

        if threshold < 2:
            return self

        low_frequency_ids: np.ndarray = self.corpus.term_frequencies_below_threshold(threshold)

        self.data = self.data[~self.data.token_id.isin(low_frequency_ids)]

        return self

    def match(self, match_tokens: List[str]) -> "CoOccurrenceHelper":

        data: pd.DataFrame = self.data

        if match_tokens:
            include_ids = self.source_token2id.find(match_tokens)
            data = data[(data.w1_id.isin(include_ids)) | (data.w2_id.isin(include_ids))]

        self.data = data

        return self

    def exclude(self, excludes: Union[IntOrStr, List[IntOrStr]]) -> "CoOccurrenceHelper":

        if not excludes:
            return self

        if isinstance(excludes, (int, str)):
            excludes = [excludes]

        fg = self.source_token2id.get

        exclude_ids = [x if isinstance(x, int) else fg(x) for x in excludes]

        data: pd.DataFrame = self.data

        data = data[(~data.w1_id.isin(exclude_ids) & ~data.w2_id.isin(exclude_ids))]

        self.data = data

        return self

    """ Unchained functions/properties follows """

    def rank(self, n_top=10, column='value') -> "CoOccurrenceHelper":

        if column not in VALUE_COLUMNS:
            raise ValueError(f"largets: expected any of {', '.join(VALUE_COLUMNS)} but found {column}")

        group_columns = [x for x in self.data.columns if x not in VALUE_COLUMNS + TOKEN_COLUMNS]

        # self.data['rank'] = self.data.groupby(group_columns)[column].rank(ascending=False) #, method='first')
        # return self.data[self.data['rank'] <= n_top] # .drop(columns='rank')

        self.data = self.data[self.data.groupby(group_columns)[column].rank(ascending=False, method='first') <= n_top]

        return self

    def largest(self, n_top=10, column='value') -> "CoOccurrenceHelper":
        group_columns = list(set(self.data_pivot_keys or []).union(set(self.corpus_pivot_keys or [])))
        largest_group = self.data.groupby(group_columns)[column].nlargest(n_top)
        self.data = self.data.loc[largest_group.index.levels[1]]
        return self

    def head(self, n_head: int) -> "CoOccurrenceHelper":

        if n_head <= 0:
            return self.data

        if len(self.data) > n_head:
            print(f"warning: only {n_head} records out of {len(self.data)} records are displayed.")

        self.data = self.data.head(n_head)

        return self

    @property
    def value(self) -> pd.DataFrame:
        return self.decode().data
