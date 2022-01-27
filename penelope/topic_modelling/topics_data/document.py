from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Set

import numpy as np
import pandas as pd
from scipy import sparse as sp

from penelope import utility as pu

from .token import filter_topic_tokens_overview

if TYPE_CHECKING:
    from .topics_data import InferredTopicsData

"""dtw = document_topic_weights"""


def to_document_id_index(document_index: pd.DataFrame) -> pd.DataFrame:
    if 'document_id' in document_index.columns:
        return document_index.set_index('document_id')
    return document_index


def overload(
    dtw: pd.DataFrame,
    document_index: pd.DataFrame,
    includes: str = None,
    ignores: str = None,
) -> pd.DataFrame:
    """Add column(s) from document index to document-topics weights `dtw`."""

    di: pd.DataFrame = to_document_id_index(document_index)

    exclude_columns: Set[str] = set(dtw.columns.tolist()) | set((ignores or '').split(','))
    include_columns: Set[str] = set(includes.split(',') if includes else di.columns)
    overload_columns: List[str] = (include_columns - exclude_columns).intersection(set(di.columns))

    odtw: pd.DataFrame = dtw.merge(
        di[overload_columns],
        left_on='document_id' if 'document_id' in dtw.columns else None,
        left_index='document_id' not in dtw.columns,
        right_index=True,
        how='inner',
    )
    return odtw


def filter_by_threshold(dtw: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Filter document-topic weights by threshold"""
    if threshold <= 0:
        return dtw
    dtw: pd.DataFrame = dtw[dtw.weight >= threshold]
    return dtw


def filter_by_data_keys(dtw: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Filter document-topic weights `dtw` by key values."""
    if not kwargs:
        return dtw
    dtw: pd.DataFrame = dtw[pu.create_mask(dtw, kwargs)]
    return dtw


def filter_by_document_index_keys(dtw: pd.DataFrame, *, document_index: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Filter document-topic weights `dtw` by attribute values in document index."""
    if len(kwargs) == 0:
        return dtw
    # document_ids: Set[int] = set(document_index[pu.create_mask(document_index, kwargs)].document_id)
    # dtw: pd.DataFrame = dtw[dtw['document_id'].isin(document_ids)]

    di: pd.DataFrame = to_document_id_index(document_index)

    document_ids: pd.DataFrame = pd.DataFrame(data=None, index=di[pu.create_mask(di, kwargs)].index)
    dtw = dtw.merge(document_ids, left_on='document_id', right_index=True, how='inner')

    return dtw


def filter_by_keys(dtw: pd.DataFrame, document_index: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Filter document-topic weights `dtw` by key values."""

    dtw_keys: dict = pu.dict_subset(kwargs, set(dtw.columns))
    dtw: pd.DataFrame = filter_by_data_keys(dtw, **dtw_keys)

    document_keys: dict = {k: v for k, v in kwargs.items() if k in document_index.columns and k not in dtw.columns}
    dtw = filter_by_document_index_keys(dtw, document_index=document_index, **document_keys)
    return dtw


def filter_by_topics(dtw: pd.DataFrame, topic_ids: Sequence[int]) -> pd.DataFrame:
    dtw: pd.DataFrame = dtw[dtw['topic_id'].isin(topic_ids)]
    return dtw


def filter_by_text(dtw: pd.DataFrame, topic_token_overview: pd.DataFrame, search_text: str, n_top: int) -> pd.DataFrame:
    if len(search_text) <= 2:
        return dtw
    topic_ids: list[int] = filter_topic_tokens_overview(
        topic_token_overview, search_text=search_text, n_top=n_top
    ).index.tolist()
    dtw: pd.DataFrame = filter_by_topics(dtw, topic_ids)
    return dtw


def filter_by_inner_join(
    dtw: pd.DataFrame, other: pd.DataFrame | pd.Series | Sequence[int], left_index: bool = True, left_on: str = None
):
    """FIXME: This is faster than `isin` for large data sets"""
    other: pd.DataFrame = pd.DataFrame(data=None, index=other.index if hasattr(other, 'index') else other)
    dtw = dtw.merge(other, left_index=left_index, left_on=left_on, right_index=True, how='inner')
    return dtw


def filter_by_n_top(dtw: pd.DataFrame, n_top: int = 500) -> pd.DataFrame:
    dtw: pd.DataFrame = (
        dtw.set_index('document_id').sort_values('weight', ascending=False).head(n_top)  # .drop(['topic_id'], axis=1)
    )
    return dtw


def compute_topic_proportions(dtw: pd.DataFrame, document_index: pd.DataFrame) -> pd.DataFrame:
    """Compute topics' proportion for all documents."""
    if 'n_tokens' not in document_index.columns:
        return None
    doc_sizes: np.ndarray = document_index.n_tokens.values
    theta: sp.coo_matrix = sp.coo_matrix((dtw.weight, (dtw.document_id, dtw.topic_id)))
    theta_mult_doc_length: np.ndarray = theta.T.multiply(doc_sizes).T
    topic_frequency: np.ndarray = theta_mult_doc_length.sum(axis=0).A1
    topic_proportion: np.ndarray = topic_frequency / topic_frequency.sum()
    return topic_proportion


class DocumentTopicsCalculator:
    def __init__(self, inferred_data: InferredTopicsData):
        self.inferred_data: InferredTopicsData = inferred_data
        self.data: pd.DataFrame = inferred_data.document_topic_weights
        self.document_index = self.inferred_data.document_index.set_index("document_id", drop=True)

    @property
    def value(self) -> pd.DataFrame:
        return self.data

    # @property
    # def document_index(self) -> pd.DataFrame:
    #     return self.inferred_data.document_index

    def copy(self) -> "DocumentTopicsCalculator":
        self.data = self.data.copy()
        return self

    def reset(self) -> "DocumentTopicsCalculator":
        self.data: pd.DataFrame = self.inferred_data.document_topic_weights
        return self

    def overload(self, includes: str = None, ignores: str = None) -> "DocumentTopicsCalculator":
        """Add column(s) from document index to data."""
        self.data = overload(self.data, self.document_index, includes=includes, ignores=ignores)
        return self

    def threshold(self, threshold: float = 0.01) -> "DocumentTopicsCalculator":
        self.data = filter_by_threshold(self.data, threshold=threshold)
        return self

    def filter_by_n_top(self, n_top: int) -> "DocumentTopicsCalculator":
        self.data = filter_by_n_top(self.data, n_top=n_top)
        return self

    def filter_by_keys(self, **kwargs) -> "DocumentTopicsCalculator":
        """Filter data by key values. Return self."""
        self.data = filter_by_keys(self.data, document_index=self.document_index, **kwargs)
        return self

    def filter_by_data_keys(self, **kwargs) -> "DocumentTopicsCalculator":
        """Filter data by key values. Return self."""
        self.data = filter_by_data_keys(self.data, **kwargs)
        return self

    def filter_by_document_keys(self, **kwargs) -> "DocumentTopicsCalculator":
        """Filter data by key values. Returnm self."""
        self.data = filter_by_document_index_keys(self.data, document_index=self.document_index, **kwargs)
        return self

    def filter_by_topics(self, topic_ids: Sequence[int]) -> "DocumentTopicsCalculator":
        self.data = self.data[self.data['topic_id'].isin(topic_ids)]
        return self

    def filter_by_text(self, search_text: str, n_top: int) -> "DocumentTopicsCalculator":
        self.data = filter_by_text(
            self.data,
            topic_token_overview=self.inferred_data.topic_token_overview,
            search_text=search_text,
            n_top=n_top,
        )
        return self

    def topic_proportions(self) -> pd.DataFrame:
        """Compute topics' proportion in entire corpus."""
        data: pd.DataFrame = compute_topic_proportions(self.data, self.inferred_data.document_index)
        return data
