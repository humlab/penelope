from typing import Any, List, Mapping, Sequence, Set

import numpy as np
import pandas as pd
from penelope import utility

from .interfaces import InferredTopicsData
from .utility import filter_topic_tokens_overview


class FilterDocumentTopicWeights:
    def __init__(
        self,
        inferred_topics: InferredTopicsData,
    ):

        self.inferred_topics: InferredTopicsData = inferred_topics
        self.data: pd.DataFrame = self.inferred_topics.document_topic_weights

    @property
    def value(self) -> pd.DataFrame:
        return self.data

    @property
    def document_index(self) -> pd.DataFrame:
        return self.inferred_topics.document_index

    def copy(self) -> "FilterDocumentTopicWeights":
        self.data = self.data.copy()
        return self

    def reset(self) -> "FilterDocumentTopicWeights":
        self.data: pd.DataFrame = self.inferred_topics.document_topic_weights
        return self

    def threshold(self, threshold: float = 0.0) -> "FilterDocumentTopicWeights":
        """Filter document-topic weights by threshold"""

        if threshold > 0:
            self.data = self.data[self.data.weight >= threshold]

        return self

    def filter_by(
        self,
        threshold: float = 0.0,
        key_values: Mapping[str, Any] = None,
        document_key_values: Mapping[str, Any] = None,
    ) -> "FilterDocumentTopicWeights":
        return self.threshold(threshold).filter_by_data_keys(key_values).filter_by_document_keys(document_key_values)

    def filter_by_keys(self, key_values: Mapping[str, Any] = None) -> "FilterDocumentTopicWeights":
        """Filter data by key values. Return self."""

        return self.filter_by_data_keys(
            {k: v for k, v in key_values.items() if k in self.data.columns}
        ).filter_by_document_keys(
            {k: v for k, v in key_values.items() if k in self.document_index.columns and k not in self.data.columns}
        )

    def filter_by_data_keys(self, key_values: Mapping[str, Any] = None) -> "FilterDocumentTopicWeights":
        """Filter data by key values. Return self."""

        if key_values is not None:

            self.data = self.data[utility.create_mask(self.data, key_values)]

        return self

    def filter_by_document_keys(self, key_values: Mapping[str, Any] = None) -> "FilterDocumentTopicWeights":
        """Filter data by key values. Returnm self."""

        if key_values is not None:

            mask: np.ndarray = utility.create_mask(self.document_index, key_values)

            document_index: pd.DataFrame = self.document_index[mask]
            document_ids: Set[int] = set(document_index.document_id)

            self.data = self.data[self.data.document_id.isin(document_ids)]

        return self

    def overload(self, includes: str = None, ignores: str = None) -> "FilterDocumentTopicWeights":
        """Add column(s) from document index to data."""
        exclude_columns: Set[str] = set(self.data.columns.tolist()) | set((ignores or '').split(','))
        include_columns: Set[str] = set(includes.split(',') if includes else self.document_index.columns)
        overload_columns: List[str] = (include_columns - exclude_columns).intersection(set(self.document_index.columns))

        self.data = self.data.merge(
            self.document_index.set_index('document_id')[overload_columns],
            left_on='document_id',
            right_index=True,
            how='inner',
        )
        return self

    def filter_by_text(self, search_text: str, n_top: int) -> "FilterDocumentTopicWeights":

        if len(search_text) > 2:

            topic_ids: List[int] = filter_topic_tokens_overview(
                self.inferred_topics.topic_token_overview, search_text=search_text, n_top=n_top
            ).index.tolist()
            self.filter_by_topics(topic_ids)

        return self

    def filter_by_topics(self, topic_ids: Sequence[int]) -> "FilterDocumentTopicWeights":
        self.data = self.data[self.data.topic_id.isin(topic_ids)]
        return self
