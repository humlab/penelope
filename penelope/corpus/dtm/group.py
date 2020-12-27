from __future__ import annotations

from typing import List, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
import scipy
from penelope.utility import dict_of_key_values_inverted_to_dict_of_value_key

from ..document_index import DocumentIndex
from .interface import IVectorizedCorpus, IVectorizedCorpusProtocol, VectorizedCorpusError

T = TypeVar("T", int, str)


class GroupByMixIn:
    def group_by_category(
        self: IVectorizedCorpusProtocol,
        column_name: str,
        *,
        aggregate: str = 'sum',
        fill_gaps: bool = False,
        fill_steps: int = 1,
    ) -> IVectorizedCorpus:
        """Groups document index by category column,

        Args:
            category_column (str): The column to group by, must exist, must be of int or str type
            transformer (callable, dict, None): Transforms to apply to column before grouping
            index_values (pd.Series, List[T]): pandas index of returned document index

        Returns
        -------
        corpus: IVectorizedCorpus
            DTM of size K where K is the number of unique categorical values in `df[column]`
            INDEX of length K with category values as DOCUMENT_NAME, where i:th value is category of i:th row in returned matrix
        """

        if column_name not in self.document_index.columns:
            raise VectorizedCorpusError(f"expected column {column_name} in index, but found no such thing")

        category_series: pd.Series = self.document_index[column_name]

        categories: Sequence[T] = _get_unique_category_values(
            category_series, fill_gaps=fill_gaps, fill_steps=fill_steps
        )

        bag_term_matrix = _group_bag_term_matrix_by_category(
            bag_term_matrix=self.bag_term_matrix,
            category_series=self.document_index[column_name],
            categories=categories,
            aggregate=aggregate,
        )

        document_index: DocumentIndex = (
            DocumentIndex(self.document_index)
            .group_by_column(column_name=column_name, index_values=categories)
            .set_strictly_increasing_index()
            .document_index
        )

        corpus: IVectorizedCorpus = self.create(
            bag_term_matrix,
            token2id=self.token2id,
            document_index=document_index,
            token_counter=self.token_counter,
        )

        return corpus

    def group_by_year(self: IVectorizedCorpusProtocol, aggregate='sum', fill_gaps=True) -> IVectorizedCorpus:
        """Returns a new corpus where documents have been grouped and summed up by year."""
        return self.group_by_category('year', aggregate=aggregate, fill_gaps=fill_gaps, fill_steps=1)

    def group_by_period(
        self: IVectorizedCorpusProtocol,
        *,
        period: Union[str, dict],
        aggregate: str = 'sum',
        fill_gaps: bool = False,
    ) -> IVectorizedCorpus:
        """Returns a new corpus where documents have been grouped and summed up by year groups.
        Adds a new column named 'lustrum', 'decade' or 'periods' (if dict)"""

        known_periods: dict = {'year': 1, 'lustrum': 5, 'decade': 10}

        if isinstance(period, str):

            if period not in known_periods:
                raise VectorizedCorpusError("unknown category specifier {category_specifier} ")

            if period == 'year':
                return self.group_by_category('year', aggregate=aggregate, fill_gaps=fill_gaps, fill_steps=1)

            categorizer = lambda x: x - int(x % known_periods[period])

        else:
            year_group_mapping = dict_of_key_values_inverted_to_dict_of_value_key(period)
            categorizer = lambda x: year_group_mapping.get(x, np.nan)
            period = 'period'

        self.document_index[period] = self.document_index.year.apply(categorizer)

        corpus = self.group_by_category(
            period,
            fill_gaps=fill_gaps,
            fill_steps=known_periods.get(period, 1),
        )

        return corpus


def _get_unique_category_values(category_series: pd.Series, fill_gaps: bool = True, fill_steps: int = 1):
    """Returns dorted distinct category values, optionally with gaps filled"""
    if fill_gaps:
        return list(range(category_series.min(), category_series.max() + 1, fill_steps))
    return list(sorted(category_series.unique().tolist()))


def _group_bag_term_matrix_by_category(
    bag_term_matrix: scipy.sparse.csr_matrix,
    *,
    category_series: pd.Series,
    categories: List[Union[int, str]],
    aggregate: str,
) -> np.ndarray:
    """Returns a new DTM where rows having same values (as specified by category_series) are grouped.

    Args:
        bag_term_matrix (scipy.sparse.csr_matrix):  Original DTM to group
        category_series (pd.Series):                Specifies category value for each row on `bag_term_matrix`
        categories (List[Union[int, str]]):         Category value domain (unique list of categories)
        aggregate (str):                            How to reduce rows in category group, `sum` (default) or mean

    Returns:
        np.ndarray: Reduced matrix
    """
    assert aggregate in {'sum', 'mean'}

    shape = (len(categories), bag_term_matrix.shape[1])
    dtype = np.int64 if np.issubdtype(bag_term_matrix.dtype, np.integer) and aggregate == 'sum' else np.float64
    grouped_bag_term_matrix = np.zeros(shape=shape, dtype=dtype)
    for i, value in enumerate(categories):
        indices = category_series[category_series == value].index.tolist()
        if len(indices) == 0:
            continue
        if aggregate == 'mean':
            grouped_bag_term_matrix[i, :] = bag_term_matrix[indices, :].mean(axis=0)
        else:
            grouped_bag_term_matrix[i, :] = bag_term_matrix[indices, :].sum(axis=0)

    return grouped_bag_term_matrix
