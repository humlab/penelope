from __future__ import annotations

from typing import Callable, List, Literal, Mapping, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import scipy
from penelope.type_alias import DocumentIndex
from scipy import sparse as sp

from penelope.utility import PropertyValueMaskingOpts

from ..document_index import (
    KNOWN_TIME_PERIODS,
    DocumentIndexHelper,
    TimePeriodSpecifier,
    create_time_period_categorizer,
)
from .interface import IVectorizedCorpus, IVectorizedCorpusProtocol, VectorizedCorpusError

T = TypeVar("T", int, str)

# pylint: disable=no-member


class GroupByMixIn:
    def group_by_pivot_column(
        self: IVectorizedCorpusProtocol,
        pivot_column_name: str,
        *,
        aggregate: str = 'sum',
        fill_gaps: bool = False,
        fill_steps: int = 1,
        target_column_name: str = 'category',
    ) -> IVectorizedCorpus:
        """Groups corpus and document index by an existing pivot column,

        Args:
            column_name (str): The column to group by, MUST EXIST, must be of int or str type
            transformer (callable, dict, None): Transforms to apply to column before grouping
            index_values (pd.Series, List[T]): pandas index of returned document index
            target_column_name (str): name of (possibly transformed) pivot column

        Returns
        -------
        corpus: IVectorizedCorpus
            DTM of size K where K is the number of unique categorical values in `df[column]`
            INDEX of length K with category values as DOCUMENT_NAME, where i:th value is category of i:th row in returned matrix
        """

        if pivot_column_name not in self.document_index.columns:
            raise VectorizedCorpusError(f"expected column {pivot_column_name} in index, but found no such thing")

        category_series: pd.Series = self.document_index[pivot_column_name]

        categories: Sequence[T] = create_category_series(category_series, fill_gaps=fill_gaps, fill_steps=fill_steps)

        bag_term_matrix = group_DTM_by_category_series(
            bag_term_matrix=self.bag_term_matrix,
            category_series=self.document_index[pivot_column_name],
            categories=categories,
            aggregate=aggregate,
        )

        document_index: DocumentIndexHelper = (
            DocumentIndexHelper(self.document_index)
            .group_by_column(
                pivot_column_name=pivot_column_name,
                index_values=categories,
                target_column_name=target_column_name,
            )
            .set_strictly_increasing_index()
            .document_index
        )

        if pivot_column_name == 'year':
            """Don't loose year if we group by year and rename target"""
            document_index['year'] = document_index[target_column_name]

        corpus: IVectorizedCorpus = self.create(
            bag_term_matrix,
            token2id=self.token2id,
            document_index=document_index,
            overridden_term_frequency=self.overridden_term_frequency,
            **self.payload,
        )

        return corpus

    def group_by_year(
        self: IVectorizedCorpusProtocol,
        aggregate='sum',
        fill_gaps=True,
        target_column_name: str = 'category',
    ) -> IVectorizedCorpus:
        """ShortCut: Groups by existing year column"""
        corpus: IVectorizedCorpus = self.group_by_pivot_column(
            pivot_column_name='year',
            aggregate=aggregate,
            fill_gaps=fill_gaps,
            fill_steps=1,
            target_column_name=target_column_name,
        )
        return corpus

    def group_by_time_period(
        self: IVectorizedCorpusProtocol,
        *,
        time_period_specifier: TimePeriodSpecifier,
        aggregate: str = 'sum',
        fill_gaps: bool = False,
        target_column_name: str = 'time_period',
    ) -> IVectorizedCorpus:
        """Groups corpus and index by new column create accordning to `time_period_specifier`.

        Args:
            self (IVectorizedCorpusProtocol): Vectorized corpus
            time_period_specifier (TimePeriodSpecifier): Specifies construction of pivot_column (categorizer)
            aggregate (str, optional): Aggregate function. Defaults to 'sum'.
            fill_gaps (bool, optional): Defaults to False.

        Returns:
            IVectorizedCorpus: [description]
        """

        if time_period_specifier == 'year':
            return self.group_by_year(
                aggregate=aggregate,
                fill_gaps=fill_gaps,
                target_column_name=target_column_name,
            )

        self.document_index[target_column_name] = self.document_index.year.apply(
            create_time_period_categorizer(time_period_specifier)
        )

        corpus = self.group_by_pivot_column(
            pivot_column_name=target_column_name,
            aggregate=aggregate,
            fill_gaps=fill_gaps,
            fill_steps=KNOWN_TIME_PERIODS.get(time_period_specifier, 1),
            target_column_name=target_column_name,
        )

        return corpus

    def group_by_time_period_optimized(
        self: IVectorizedCorpusProtocol,
        time_period_specifier: Union[str, dict],
        aggregate: str = 'sum',
        fill_gaps: bool = False,
        target_column_name: str = 'time_period',
    ) -> IVectorizedCorpus:
        """Groups corpus by specified time_period_specifier.
        Uses scipy sparse lil format during construction.
        Args:
            time_period_specifier (Union[str, dict]): [description]
            aggregate (str, optional): [description]. Defaults to 'sum'.

        Returns:
            IVectorizedCorpus: grouped corpus
        """

        if time_period_specifier == 'year':
            return self.group_by_year(
                aggregate=aggregate,
                fill_gaps=fill_gaps,
                target_column_name=target_column_name,
            )

        if fill_gaps:
            raise NotImplementedError("group_by_time_period_optimized: fill gaps when specifier not year")

        document_index, category_indices = DocumentIndexHelper(self.document_index).group_by_time_period(
            time_period_specifier=time_period_specifier,
            target_column_name=target_column_name,
        )

        matrix: scipy.sparse.spmatrix = group_DTM_by_category_indices_mapping(
            bag_term_matrix=self.bag_term_matrix,
            category_indices=category_indices,
            aggregate=aggregate,
            document_index=document_index,
            pivot_column_name=target_column_name,
        )

        grouped_corpus: IVectorizedCorpus = self.create(
            matrix.tocsr(),
            token2id=self.token2id,
            document_index=document_index,
            overridden_term_frequency=self.overridden_term_frequency,
            **self.payload,
        )
        return grouped_corpus

    def group_by_indices_mapping(
        self: IVectorizedCorpusProtocol,
        document_index: pd.DataFrame,
        category_indices: Mapping[int, List[int]],
        aggregate: str = 'sum',
        dtype: np.dtype = None,
    ) -> IVectorizedCorpus:
        """Groups corpus by index mapping

        Args:
            document_index (pd.DataFrame): Grouped document index
            category_indices (Mapping[int, List[int]]): [description]
            aggregate (str, optional): [description]. Defaults to 'sum'.
            dtype (np.dtype, optional): [description]. Defaults to None.

        Returns:
            IVectorizedCorpus: [description]
        """
        matrix: scipy.sparse.spmatrix = group_DTM_by_indices_mapping(
            dtm=self.bag_term_matrix,
            n_docs=len(document_index),
            category_indices=category_indices,
            aggregate=aggregate,
            dtype=dtype,
        )
        grouped_corpus: IVectorizedCorpus = self.create(
            matrix.tocsr(),
            token2id=self.token2id,
            document_index=document_index,
            overridden_term_frequency=self.overridden_term_frequency,
            **self.payload,
        )
        return grouped_corpus

    def group_by_pivot_keys(
        self: IVectorizedCorpusProtocol,
        temporal_key: Literal['year', 'decade', 'lustrum'],
        pivot_keys: List[str],
        pivot_keys_filter: PropertyValueMaskingOpts,
        document_namer: Callable[[pd.DataFrame], pd.Series],
        count_column: str = 'n_tokens',
        aggregate: str = 'sum',
        dtype: np.dtype = None,
    ):
        """Groups corpus by a temporal key and zero to many pivot keys

        Args:
            self (IVectorizedCorpusProtocol): [description]
            temporal_key (Literal['year', 'decade', 'lustrum']): Temporal grouping key value (year, lustrum, decade)
            pivot_keys (List[str]): Grouping key value, must be discrete categorical values.
            pivot_keys_filter (PropertyValueMaskingOpts): Filters that should be applied to documets index.
            document_namer (Callable[[pd.DataFrame], pd.Series]): Funciton that computes a document name for each result groups.
            count_column (str, optional): Token count column in document index that should be aggregated. Defaults to 'n_tokens'.
            aggregate (str, optional): Aggregate function for DTM and document index (n_tokens). Defaults to 'sum'.
            dtype (np.dtype, optional): Value type of target DTM matrix. Defaults to None.
        """

        def default_document_namer(df: pd.DataFrame) -> pd.Series:
            return df[[temporal_key] + pivot_keys].apply(lambda x: '_'.join([str(t) for t in x]), axis=1)

        if document_namer is None:
            document_namer = default_document_namer

        fdi: pd.DataFrame = (
            self.document_index
            if not pivot_keys or pivot_keys_filter is None or len(pivot_keys_filter) == 0
            else self.document_index[pivot_keys_filter.mask[self.document_index]]
        )
        gdi: pd.DataFrame = (
            fdi.groupby([temporal_key] + pivot_keys).agg({'document_id': list, count_column: aggregate}).reset_index()
        )

        gdi['document_ids'] = gdi.document_id
        gdi['document_id'] = gdi.index
        gdi['document_name'] = document_namer(gdi)
        gdi['filename'] = gdi.document_name

        category_indices: Mapping[int, List[int]] = gdi.document_ids.to_dict()

        return self.group_by_indices_mapping(
            document_index=gdi,
            category_indices=category_indices,
            aggregate=aggregate,
            dtype=dtype,
        )


def group_DTM_by_category_indices_mapping(
    *,
    bag_term_matrix: scipy.sparse.spmatrix,
    category_indices: Mapping[int, List[int]],
    aggregate: str,
    document_index: DocumentIndex,
    pivot_column_name: str,
):
    shape = (len(document_index), bag_term_matrix.shape[1])
    dtype = np.int32 if np.issubdtype(bag_term_matrix.dtype, np.integer) and aggregate == 'sum' else np.float64
    matrix: scipy.sparse.lil_matrix = scipy.sparse.lil_matrix(shape, dtype=dtype)

    for document_id, category_value in document_index[pivot_column_name].to_dict().items():
        indices = category_indices.get(category_value, [])
        if len(indices) == 0:
            continue
        matrix[document_id, :] = (
            bag_term_matrix[indices, :].mean(axis=0) if aggregate == 'mean' else bag_term_matrix[indices, :].sum(axis=0)
        )

    return matrix


def create_category_series(category_series: pd.Series, fill_gaps: bool = True, fill_steps: int = 1):
    """Returns sorted distinct category values, optionally with gaps filled"""
    if fill_gaps:
        return list(range(category_series.min(), category_series.max() + 1, fill_steps))
    return list(sorted(category_series.unique().tolist()))


def group_DTM_by_category_series(
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
    matrix = np.zeros(shape=shape, dtype=dtype)
    for i, value in enumerate(categories):
        indices = category_series[category_series == value].index.tolist()
        if len(indices) == 0:
            continue
        if aggregate == 'mean':
            matrix[i, :] = bag_term_matrix[indices, :].mean(axis=0)
        else:
            matrix[i, :] = bag_term_matrix[indices, :].sum(axis=0)

    return matrix


def group_DTM_by_indices_mapping(
    dtm: sp.spmatrix,
    n_docs: int,
    category_indices: Mapping[int, List[int]],
    aggregate: str = 'sum',
    dtype: np.dtype = None,
):

    shape: Tuple[int, int] = (n_docs, dtm.shape[1])

    dtype: np.dtype = dtype or (np.int32 if np.issubdtype(dtm.dtype, np.integer) and aggregate == 'sum' else np.float64)

    matrix: sp.lil_matrix = sp.lil_matrix(shape, dtype=dtype)

    if aggregate == 'mean':

        for document_id, indices in category_indices.items():
            if len(indices) > 0:
                matrix[document_id, :] = dtm[indices, :].mean(axis=0)
    else:

        for document_id, indices in category_indices.items():
            if len(indices) > 0:
                matrix[document_id, :] = dtm[indices, :].sum(axis=0)

    return matrix
