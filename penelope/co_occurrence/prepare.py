from typing import List, Mapping, Union

import numpy as np
import pandas as pd

from penelope.common.keyness import KeynessMetric, partitioned_significances
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.utility import do_not_use

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
        co_occurrences: pd.DataFrame = None,
    ):

        self.corpus: VectorizedCorpus = corpus
        self.source_token2id: Token2Id = source_token2id
        self.corpus_pivot_keys: List[str] = [pivot_keys] if isinstance(pivot_keys, str) else pivot_keys

        self.co_occurrences: pd.DataFrame = (
            co_occurrences if co_occurrences is not None else self.corpus.to_co_occurrences(source_token2id)
        )
        self.data: pd.DataFrame = None
        self.data_pivot_keys: List[str] = None

        self.reset()

    def reset(self) -> "CoOccurrenceHelper":

        self.data: pd.DataFrame = self.co_occurrences  # .copy()
        self.data_pivot_keys = self.corpus_pivot_keys

        if 'time_period' not in self.data.columns:
            if 'year' in self.data.columns:
                self.data['time_period'] = self.data.year

        return self

    def normalize(
        self, pivot_keys: Union[str, List[str]], taget_name='value_n_t', normalize_key: str = 'n_raw_tokens'
    ) -> "CoOccurrenceHelper":

        self.data[taget_name] = self._normalize(self.data, pivot_keys=pivot_keys, normalize_key=normalize_key)

        return self

    def _normalize(
        self, data: pd.DataFrame, pivot_keys: Union[str, List[str]], normalize_key: str = 'n_raw_tokens'
    ) -> "CoOccurrenceHelper":
        """Normalizes groups defined by `pivot_keys` by raw token counts (given by document index)"""

        missing_columns: List[str] = [
            x for x in (pivot_keys + [normalize_key]) if x not in self.corpus.document_index.columns
        ]
        if len(missing_columns) > 0:
            raise f"BugCheck: expected {','.join(missing_columns)} to be corpus' document index."

        series: pd.Series = (
            data.value
            / pd.merge(
                data[pivot_keys],
                self.corpus.document_index.groupby(pivot_keys)[normalize_key].sum(),
                left_on=pivot_keys,
                right_index=True,
            )[normalize_key]
        )

        return series

    @do_not_use
    def groupby(
        self,
        document_pivot_keys: Union[str, List[str]],
        normalize: bool = False,
        normalize_key: str = 'n_raw_tokens',
        target_pivot_key='time_period',
    ) -> "CoOccurrenceHelper":
        """Groups co-occurrences data frame by given document index properties (i.e. year)
        Extends/overloads the co-occurrence data frame with specified keys
        Adds a `target_pivot_key` column as an alias from added column"""

        if self.data_pivot_keys:
            raise ValueError("Already grouped, please reset before calling again")

        if not document_pivot_keys:
            raise ValueError("pivot keys is not specified")

        data: pd.DataFrame = self.data

        if isinstance(document_pivot_keys, str):
            document_pivot_keys = [document_pivot_keys]

        document_pivot_keys = [
            g for g in document_pivot_keys if g in self.corpus.document_index.columns and g not in data.columns
        ]

        if len(document_pivot_keys) == 0:
            raise ValueError("No keys to group by!")

        """Add grouping columns to data"""
        data: pd.DataFrame = data.merge(
            self.corpus.document_index[document_pivot_keys], left_on='document_id', right_index=True, how='inner'
        )

        """Group and sum up data"""
        data = data.groupby(document_pivot_keys + ID_COLUMNS)['value'].sum().reset_index()

        """Divide window counts with time-periods token counts"""
        if normalize:
            data['value_n_t'] = self._normalize(data, pivot_keys=document_pivot_keys, normalize_key=normalize_key)

        if target_pivot_key not in data.columns:
            """Only handles single keys!"""
            assert document_pivot_keys[0] in data.columns
            data[target_pivot_key] = data[document_pivot_keys[0]]

        self.data = data
        self.data_pivot_keys = document_pivot_keys

        return self

    def decode(self) -> "CoOccurrenceHelper":

        self.data = decode_tokens(
            self.data,
            self.source_token2id.id2token,
            self.corpus.id2token,
            copy=True,
        )
        return self

    def trunk_by_global_count(self, threshold: int) -> "CoOccurrenceHelper":

        if len(self.data) == 0:
            return self

        if threshold < 2:
            return self

        low_frequency_ids: np.ndarray = self.corpus.term_frequencies_greater_than_or_equal_to_threshold(threshold)

        self.data = self.data[~self.data.token_id.isin(low_frequency_ids)]

        return self

    def match(self, match_tokens: List[str]) -> "CoOccurrenceHelper":

        if len(self.data) == 0:
            return self

        data: pd.DataFrame = self.data

        if match_tokens:
            include_ids = self.source_token2id.find(match_tokens)
            data = data[(data.w1_id.isin(include_ids)) | (data.w2_id.isin(include_ids))]

        self.data = data

        return self

    def exclude(self, excludes: Union[IntOrStr, List[IntOrStr]]) -> "CoOccurrenceHelper":

        if len(self.data) == 0:
            return self

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

    def rank(self, n_top: int = 10, column: str = 'value') -> "CoOccurrenceHelper":

        if len(self.data) == 0:
            return self

        if column not in VALUE_COLUMNS:
            raise ValueError(f"largets: expected any of {', '.join(VALUE_COLUMNS)} but found {column}")

        group_columns: List[str] = [x for x in self.data.columns if x not in VALUE_COLUMNS + TOKEN_COLUMNS]

        # self.data['rank'] = self.data.groupby(group_columns)[column].rank(ascending=False) #, method='first')
        # return self.data[self.data['rank'] <= n_top] # .drop(columns='rank')

        self.data = self.data[self.data.groupby(group_columns)[column].rank(ascending=False, method='first') <= n_top]

        return self

    def largest(self, n_top: int = 10, column: str = 'value') -> "CoOccurrenceHelper":

        if len(self.data) == 0:
            return self

        group_columns = list(set(self.data_pivot_keys or []).union(set(self.corpus_pivot_keys or [])))

        if not group_columns:
            raise ValueError("fatal: group  columns cannot be empty")

        largest_indices = self.data.groupby(group_columns)[column].nlargest(n_top).index.get_level_values(-1).tolist()

        self.data = self.data.loc[largest_indices]

        return self

    def head(self, n_head: int) -> "CoOccurrenceHelper":

        if len(self.data) == 0:
            return self

        if n_head <= 0:
            return self.data

        if len(self.data) > n_head:
            print(f"warning: only {n_head} records out of {len(self.data)} records are displayed.")

        self.data = self.data.head(n_head)

        return self

    def weigh_by_significance(
        self,
        co_occurrences: pd.DataFrame,
        keyness: KeynessMetric,
        pivot_key: str,
        vocabulary_size: int,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Computes statistical signficance of for co-occurrences
        Note: Compute on non-filtered co-occurrences data!
        """
        weighed_co_occurrences = partitioned_significances(
            co_occurrences=co_occurrences,
            pivot_key=pivot_key,
            keyness_metric=keyness,
            document_index=self.corpus.document_index,
            vocabulary_size=vocabulary_size,
            normalize=normalize,
        )
        self.data = weighed_co_occurrences
        return self

    @property
    def value(self) -> pd.DataFrame:
        if self.data.index.name in self.data.columns:
            self.data.rename_axis('', inplace=True)
        return self.decode().data


def decode_tokens(
    co_occurrences: pd.DataFrame,
    id2token: Mapping[int, str],
    coo_id2token: Mapping[int, str],
    copy: bool = True,
) -> pd.DataFrame:

    if 'w1' in co_occurrences.columns and 'token' in co_occurrences.columns:
        return co_occurrences

    if copy:
        co_occurrences = co_occurrences.copy()

    fg = id2token.get
    co_occurrences["w1"] = co_occurrences.w1_id.apply(fg)
    co_occurrences["w2"] = co_occurrences.w2_id.apply(fg)

    fg = coo_id2token.get
    co_occurrences["token"] = co_occurrences.token_id.apply(fg)

    return co_occurrences
