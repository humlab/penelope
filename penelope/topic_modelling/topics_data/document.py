from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Set

import numpy as np
import pandas as pd
from scipy import sparse as sp

from penelope import utility as pu

from . import prevelance
from .token import filter_topic_tokens_overview

if TYPE_CHECKING:
    from .topics_data import InferredTopicsData

"""dtw = document_topic_weights"""


def set_id_index(document_index: pd.DataFrame) -> pd.DataFrame:
    if 'document_id' in document_index.columns:
        return document_index.set_index('document_id')
    return document_index


def overload(
    *,
    dtw: pd.DataFrame,
    document_index: pd.DataFrame,
    includes: str = None,
    ignores: str = None,
) -> pd.DataFrame:
    """Add column(s) from document index to document-topics weights `dtw`."""

    di: pd.DataFrame = set_id_index(document_index)

    exclude_columns: Set[str] = set(dtw.columns.tolist()) | set((ignores or '').split(','))
    include_columns: Set[str] = set(includes.split(',') if includes else di.columns)
    overload_columns: List[str] = list((include_columns - exclude_columns).intersection(set(di.columns)))

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

    di: pd.DataFrame = set_id_index(document_index)

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
    def __init__(self, inferred_topics: InferredTopicsData):

        self.inferred_topics: InferredTopicsData = inferred_topics
        self.data: pd.DataFrame = inferred_topics.document_topic_weights

    @property
    def value(self) -> pd.DataFrame:
        return self.data

    @property
    def document_index(self) -> pd.DataFrame:
        return self.inferred_topics.document_index

    def copy(self) -> "DocumentTopicsCalculator":
        self.data = self.data.copy()
        return self

    def reset(self, data: pd.DataFrame = None) -> "DocumentTopicsCalculator":
        self.data: pd.DataFrame = data if data is not None else self.inferred_topics.document_topic_weights
        return self

    def overload(self, includes: str = None, ignores: str = None) -> "DocumentTopicsCalculator":
        """Add column(s) from document index to data."""
        self.data = overload(dtw=self.data, document_index=self.document_index, includes=includes, ignores=ignores)
        return self

    def threshold(self, threshold: float = 0.01) -> "DocumentTopicsCalculator":
        self.data = filter_by_threshold(self.data, threshold=threshold)
        return self

    def head(self, n: int) -> "DocumentTopicsCalculator":
        self.data = self.data.head(n)
        return self

    def filter_by_n_top(self, n_top: int) -> "DocumentTopicsCalculator":
        self.data = filter_by_n_top(self.data, n_top=n_top)
        return self

    def filter_by_keys(self, **kwargs) -> "DocumentTopicsCalculator":
        """Filter data by key values."""
        if kwargs:
            self.data = filter_by_keys(self.data, document_index=self.document_index, **kwargs)
        return self

    def filter_by_data_keys(self, **kwargs) -> "DocumentTopicsCalculator":
        """Filter data by key values."""
        if kwargs:
            self.data = filter_by_data_keys(self.data, **kwargs)
        return self

    def filter_by_document_keys(self, **kwargs) -> DocumentTopicsCalculator:
        """Filter data by key values."""
        if kwargs:
            self.data = filter_by_document_index_keys(self.data, document_index=self.document_index, **kwargs)
        return self

    def filter_by_topics(self, topic_ids: Sequence[int], negate: bool = False) -> "DocumentTopicsCalculator":
        if topic_ids:
            mask = self.data['topic_id'].isin(topic_ids)
            self.data = self.data[(mask if not negate else ~mask)]
        return self

    def filter_by_focus_topics(self, topic_ids: Sequence[int]) -> "DocumentTopicsCalculator":
        if topic_ids:
            df_focus: pd.DataFrame = self.data[self.data.topic_id.isin(topic_ids)].set_index("document_id")
            df_others: pd.DataFrame = self.data[~self.data.topic_id.isin(topic_ids)].set_index("document_id")
            df_others = df_others[df_others.index.isin(df_focus.index)]
            self.data: pd.DataFrame = df_focus.append(df_others).reset_index()
        return self

    def filter_by_text(self, search_text: str, n_top: int) -> "DocumentTopicsCalculator":
        self.data = filter_by_text(
            self.data,
            topic_token_overview=self.inferred_topics.topic_token_overview,
            search_text=search_text,
            n_top=n_top,
        )
        return self

    def topic_proportions(self) -> pd.DataFrame:
        """Compute topics' proportion in entire corpus."""
        data: pd.DataFrame = compute_topic_proportions(self.data, self.document_index)
        return data

    def yearly_topic_weights(
        self,
        result_threshold: float,
        n_top_relevance: int,
        topic_ids: None | int | list[int] = None,
    ) -> "DocumentTopicsCalculator":
        self.data = prevelance.compute_yearly_topic_weights(
            self.data,
            document_index=self.document_index,
            threshold=result_threshold or 0,
            n_top_relevance=n_top_relevance,
            topic_ids=topic_ids,
        )
        return self

    def to_topic_topic_network(
        self,
        n_docs: int,
        pivot_keys: list[str] | str = None,
        topic_labels: dict[int, str] = True,
    ) -> DocumentTopicsCalculator:

        pivot_keys = pivot_keys or []

        data: pd.DataFrame = self.data.set_index('document_id')

        topic_product: pd.DataFrame = data.merge(data, left_index=True, right_index=True)
        topic_product = topic_product[(topic_product.topic_id_x < topic_product.topic_id_y)]

        network_data: pd.DataFrame = topic_product.groupby(
            pivot_keys + [topic_product.topic_id_x, topic_product.topic_id_y]
        ).agg(
            n_docs=('year_x', 'size')  # , weight_x=('weight_x', 'sum'), weight_y=('weight_y', 'sum')
        )

        network_data.reset_index(inplace=True)
        # network_data['weight'] = network_data.weight_x + network_data.weight_y
        # network_data.drop(columns=['weight_x', 'weight_y'], inplace=True)
        network_data.columns = pivot_keys + ['source', 'target', 'n_docs']  # , 'weight']

        # FIXME: Måste normalisera efter antal dokument!!!
        if n_docs > 1:
            network_data = network_data[network_data.n_docs >= n_docs]

        if topic_labels is not None:
            network_data['source'] = network_data['source'].apply(topic_labels.get)
            network_data['target'] = network_data['target'].apply(topic_labels.get)

        self.data = network_data

        return self

    def to_pivot_topic_network(
        self,
        *,
        pivot_key_id: str,
        pivot_key_name: str = "category",
        pivot_key_map: dict[int, str],
        aggregate: str,
        threshold: float,
        topic_labels: dict[int, str] = True,
    ) -> DocumentTopicsCalculator:

        data: pd.DataFrame = self.data  # .set_index('document_id')

        network_data: pd.DataFrame = (
            data.groupby([pivot_key_id, 'topic_id']).agg([np.mean, np.max])['weight'].reset_index()
        )
        network_data.columns = [pivot_key_id, 'topic_id', 'mean', 'max']
        network_data = network_data[(network_data[aggregate] > threshold)].reset_index()

        if len(network_data) == 0:
            return network_data

        network_data[aggregate] = pu.clamp_values(list(network_data[aggregate]), (0.1, 1.0))  # type: ignore
        network_data[pivot_key_name] = network_data[pivot_key_id].apply(pivot_key_map.get)
        network_data['weight'] = network_data[aggregate]
        network_data.drop(columns=['mean', 'max'], inplace=True)

        if topic_labels is not None:
            network_data['topic_id'] = network_data['topic_id'].apply(topic_labels.get)

        self.data = network_data

        return self
