import abc
from typing import Any, List, Mapping, Sequence, Set

import numpy as np
import pandas as pd
from penelope import utility

from .interfaces import InferredTopicsData
from .utility import filter_topic_tokens_overview


class DocumentTopicWeightsReducer:
    def __init__(
        self,
        inferred_topics: InferredTopicsData,
    ):

        self.inferred_topics: InferredTopicsData = inferred_topics
        self.data: pd.DataFrame = self.inferred_topics.document_topic_weights

    @property
    def copy(self) -> pd.DataFrame:
        return self.data.copy()

    @property
    def value(self) -> pd.DataFrame:
        return self.data

    def threshold(self, threshold: float = 0.0) -> "DocumentTopicWeightsReducer":
        """Filter document-topic weights by threshold"""

        if threshold > 0:
            self.data = self.data[self.data.weight >= threshold]

        return self

    def filter_by(
        self,
        threshold: float = 0.0,
        key_values: Mapping[str, Any] = None,
        document_key_values: Mapping[str, Any] = None,
    ) -> "DocumentTopicWeightsReducer":
        return self.threshold(threshold).filter_by_data_keys(key_values).filter_by_document_keys(document_key_values)

    def filter_by_keys(self, key_values: Mapping[str, Any] = None) -> "DocumentTopicWeightsReducer":
        """Filter data by key values. Return self."""

        return self.filter_by_data_keys(
            {k: v for k, v in key_values.items() if k in self.data.columns}
        ).filter_by_document_keys(
            {k: v for k, v in key_values.items() if k in self.document_index.columns and k not in self.data.columns}
        )

    def filter_by_data_keys(self, key_values: Mapping[str, Any] = None) -> "DocumentTopicWeightsReducer":
        """Filter data by key values. Return self."""

        if key_values is not None:

            self.data = self.data[utility.create_mask(self.data, key_values)]

        return self

    def filter_by_document_keys(self, key_values: Mapping[str, Any] = None) -> "DocumentTopicWeightsReducer":
        """Filter data by key values. Returnm self."""

        if key_values is not None:

            mask: np.ndarray = utility.create_mask(self.document_index, key_values)

            document_index: pd.DataFrame = self.document_index[mask]
            document_ids: Set[int] = set(document_index.document_id)

            self.data = self.data[self.data.document_id.isin(document_ids)]

        return self

    def overload(self, includes: str = None, ignores: str = 'year,document_name') -> "DocumentTopicWeightsReducer":

        extra_columns: List[str] = (
            includes.split(',') if includes else [x for x in self.document_index.columns.tolist()]
        )
        extra_columns = [c for c in extra_columns if c not in self.data.columns and c not in (ignores or '').split(',')]
        self.data = self.data.merge(
            self.document_index[extra_columns], left_on='document_id', right_on='document_id', how='inner'
        )
        return self

    def filter_by_text(self, search_text: str, n_top: int) -> "DocumentTopicWeightsReducer":

        if len(search_text) > 2:

            topic_ids: List[int] = filter_topic_tokens_overview(
                self.inferred_topics.topic_token_overview, search_text=search_text, n_count=n_top
            ).index.tolist()
            self.filter_by_topics(topic_ids)

        return self

    def filter_by_topics(self, topic_ids: Sequence[int]) -> "DocumentTopicWeightsReducer":
        self.data = self.data[self.data.topic_id.isin(topic_ids)]
        return self
