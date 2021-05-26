from typing import List, Union

import pandas as pd
from penelope.corpus import DocumentIndex, Token2Id
from penelope.type_alias import CoOccurrenceDataFrame

IntOrStr = Union[int, str]


class CoOccurrenceHelper:
    def __init__(
        self,
        co_occurrences: CoOccurrenceDataFrame,
        token2id: Token2Id,
        document_index: DocumentIndex,
        pivot_keys: List[str] = None,
    ):

        self.co_occurrences: CoOccurrenceDataFrame = co_occurrences
        self.token2id: Token2Id = token2id
        self.document_index: DocumentIndex = document_index
        self.data: pd.DataFrame = self.co_occurrences  # .copy()
        self.pivot_keys: List[str] = pivot_keys

    def reset(self) -> "CoOccurrenceHelper":
        self.data: pd.DataFrame = self.co_occurrences  # .copy()
        self.pivot_keys = None
        return self

    def groupby(self, keys: Union[str, List[str]]) -> "CoOccurrenceHelper":

        if not keys:
            raise ValueError("group keys is not specified")

        data: pd.DataFrame = self.data

        if isinstance(keys, str):
            keys = [keys]

        document_index: DocumentIndex = self.document_index.set_index('document_id')

        keys = [g for g in keys if g in document_index.columns and g not in data.columns]

        if len(keys) == 0:
            raise ValueError(f"group keys {' '.join(keys)} not found in document index")
        # counter_columns = [c for c in ['n_tokens', 'n_raw_tokens'] if c in document_index.columns]

        """Add grouping columns to data"""
        data = data.merge(document_index[keys], left_on='document_id', right_index=True, how='inner')

        """Group and sum up data"""
        data = data.groupby(keys + ['w1_id', 'w2_id'])['value'].sum().reset_index()

        """Divide yearly window counts with yearly token counts"""
        normalize_key = 'n_raw_tokens'
        data['value_n_t'] = data.value / pd.merge(
            data[keys],
            document_index.groupby(keys)[normalize_key].sum(),  # Yearly token counts
            left_on='year',
            right_index=True,
        )[normalize_key]

        self.data = data
        self.pivot_keys = keys

        return self

    def decode(self) -> "CoOccurrenceHelper":

        if 'w1' in self.data.columns:
            return self

        fg = self.token2id.id2token.get
        self.data["w1"] = self.data.w1_id.apply(fg)
        self.data["w2"] = self.data.w2_id.apply(fg)
        self.data["token"] = self.data.w1 + '/' + self.data.w2

        return self

    def trunk_by_global_count(self, threshold: int) -> "CoOccurrenceHelper":

        if threshold < 2:
            return self

        global_tokens_counts: pd.Series = self.co_occurrences.groupby(['token'])['value'].sum()
        threshold_tokens: pd.Index = global_tokens_counts[global_tokens_counts >= threshold].index
        self.data = self.data.set_index('token').loc[threshold_tokens]  # [['year', 'value', 'value_n_t']]

        return self

    def match(self, match_tokens: List[str]) -> "CoOccurrenceHelper":

        data: pd.DataFrame = self.data

        if match_tokens:
            include_ids = self.token2id.find(match_tokens)
            data = data[(data.w1_id.isin(include_ids)) | (data.w2_id.isin(include_ids))]

        self.data = data

        return self

    def exclude(self, excludes: Union[IntOrStr, List[IntOrStr]]) -> "CoOccurrenceHelper":

        if not excludes:
            return self

        if isinstance(excludes, (int, str)):
            excludes = [excludes]

        fg = self.token2id.get

        exclude_ids = [x if isinstance(x, int) else fg(x) for x in excludes]

        data: pd.DataFrame = self.data

        data = data[(~data.w1_id.isin(exclude_ids) & ~data.w2_id.isin(exclude_ids))]

        self.data = data

        return self

    """ Unchained functions/properties follows """

    def rank(self, n_top=10, column='value') -> "CoOccurrenceHelper":

        value_columns: List[str] = ['value', 'n_tokens', 'n_raw_tokens']
        token_columns: List[str] = ['w1', 'w2', 'token']

        if column not in value_columns:
            raise ValueError(f"largets: expected any of {', '.join(value_columns)} but found {column}")

        group_columns = [x for x in self.data.columns if x not in value_columns + token_columns]

        # self.data['rank'] = self.data.groupby(group_columns)[column].rank(ascending=False) #, method='first')
        # return self.data[self.data['rank'] <= n_top] # .drop(columns='rank')

        self.data = self.data[self.data.groupby(group_columns)[column].rank(ascending=False, method='first') <= n_top]

        return self

    def largest(self, n_top=10, column='value') -> "CoOccurrenceHelper":

        if self.pivot_keys is None:
            return self.data.sort_values(by=column, ascending=True).head(n_top)

        self.data = self.data.loc[self.data.groupby(self.pivot_keys)[column].nlargest(n_top).reset_index().level_1]

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
        return self.data
