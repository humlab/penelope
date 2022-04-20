from __future__ import annotations

import abc
import itertools
from typing import TYPE_CHECKING, Literal, NamedTuple, get_args

import numpy as np
import pandas as pd

from penelope import utility as pu

if TYPE_CHECKING:
    from .topics_data import InferredTopicsData


def default_calculator():
    return MemoizedTopicPrevalenceOverTimeCalculator(calculator=AverageTopicPrevalenceOverTimeCalculator())


class TopicPrevalenceOverTimeCalculator(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        *,
        inferred_topics: InferredTopicsData,
        filters: dict,
        threshold: float = 0.0,
        result_threshold: float = 0.0,
        n_top_relevance: int = None,
    ) -> pd.DataFrame:
        ...


class AverageTopicPrevalenceOverTimeCalculator(TopicPrevalenceOverTimeCalculator):
    def compute(
        self,
        *,
        inferred_topics: InferredTopicsData,
        filters: dict,
        threshold: float = 0.0,
        result_threshold: float = 0.0,
        n_top_relevance: int = None,
    ) -> pd.DataFrame:
        dtw: pd.DataFrame = (
            inferred_topics.calculator.reset().threshold(threshold or 0).filter_by_keys(**(filters or {})).value
        )
        if len(dtw) == 0:
            return None
        return self.compute_yearly_topic_weights(
            dtw,
            document_index=inferred_topics.document_index,
            threshold=result_threshold or 0,
            n_top_relevance=n_top_relevance,
        )

    @staticmethod
    def compute_yearly_topic_weights(
        document_topic_weights: pd.DataFrame,
        *,
        document_index: pd.DataFrame,
        threshold: float = None,
        n_top_relevance: int = None,
    ) -> pd.DataFrame:
        return compute_yearly_topic_weights(
            document_topic_weights, document_index=document_index, threshold=threshold, n_top_relevance=n_top_relevance
        )


# class RollingAverageTopicPrevalenceOverTimeCalculator(AverageTopicPrevalenceOverTimeCalculator):
#     """Not implemented"""


# class TopTopicPrevalenceOverTimeCalculator(TopicPrevalenceOverTimeCalculator):
#     """Not implemented"""


class MemoizedTopicPrevalenceOverTimeCalculator(TopicPrevalenceOverTimeCalculator):
    """Proxy calculator that returns last calculation if arguments are the same"""

    class ArgsMemory(NamedTuple):
        inferred_topics: InferredTopicsData
        filters: dict
        threshold: float = 0.0
        result_threshold: float = 0.0
        n_top_relevance: int = 0

        def validate(
            self,
            inferred_topics: InferredTopicsData,
            filters: dict,
            threshold: float = 0.0,
            result_threshold: float = 0.0,
            n_top_relevance: int = None,
        ):
            return (
                self.inferred_topics is inferred_topics
                and self.filters == filters
                and self.threshold == threshold
                and self.result_threshold == result_threshold
                and self.n_top_relevance == n_top_relevance
            )

    def __init__(self, calculator: TopicPrevalenceOverTimeCalculator):

        self.calculator: TopicPrevalenceOverTimeCalculator = calculator or AverageTopicPrevalenceOverTimeCalculator()
        self.data: pd.DataFrame = None
        self.args: MemoizedTopicPrevalenceOverTimeCalculator.ArgsMemory = None

    def compute(
        self,
        *,
        inferred_topics: InferredTopicsData,
        filters: dict,
        threshold: float = 0.0,
        result_threshold: float = 0.0,
        n_top_relevance: int = None,
    ) -> pd.DataFrame:

        if not (self.args and self.args.validate(inferred_topics, filters, threshold, result_threshold)):

            self.data = self.calculator.compute(
                inferred_topics=inferred_topics, filters=filters, threshold=threshold, result_threshold=result_threshold
            )
            self.args = MemoizedTopicPrevalenceOverTimeCalculator.ArgsMemory(
                inferred_topics=inferred_topics,
                filters=filters,
                threshold=threshold,
                result_threshold=result_threshold,
                n_top_relevance=n_top_relevance,
            )

        return self.data


DefaultPrevalenceOverTimeCalculator = MemoizedTopicPrevalenceOverTimeCalculator


def _compute_average_yearly_topic_weights_above_threshold(
    document_topic_weights: pd.DataFrame, threshold: float, target_name: str = 'average_weight'
) -> pd.DataFrame:
    """Compute average weights ignoring values below `threshold`."""
    yearly_weights: pd.DataFrame = (
        document_topic_weights[document_topic_weights.weight >= threshold]
        .groupby(['year', 'topic_id'])
        .agg(**{target_name: ('weight', np.mean)})
    )
    return yearly_weights


def _compute_yearly_topic_weights_statistics(
    dtw: pd.DataFrame,
    topic_ids: None | int | list[int] = None,
) -> pd.DataFrame:
    """Setup all topic-year combinations, aggregate max, sum, average & count."""

    if len(dtw) == 0:
        raise pu.EmptyDataError()

    year_range = range(dtw.year.min(), dtw.year.max() + 1)
    topic_ids = (
        range(0, dtw.topic_id.max() + 1)
        if topic_ids is None
        else [topic_ids]
        if isinstance(topic_ids, int)
        else topic_ids
    )

    year_topics: list[tuple[int, int]] = list(itertools.product(year_range, topic_ids))
    ytw: pd.DataFrame = (
        pd.DataFrame(year_topics, columns=['year', 'topic_id'])
        .set_index(['year', 'topic_id'])
        .join(
            dtw.groupby(['year', 'topic_id'])['weight'].agg([np.max, np.sum, np.mean, len]),
            how='left',
        )
        .fillna(0)
        .pipe(pu.rename_columns, columns=['max_weight', 'sum_weight', 'avg_weight', 'n_topic_documents'])
    )
    ytw['average_weight'] = ytw['avg_weight']
    return ytw


def _add_average_yearly_topic_weight_above_threshold(
    yearly_weights: pd.DataFrame,
    dtw: pd.DataFrame,
    threshold: float = None,
    target_name: str = 'average_weight',
) -> pd.DataFrame:
    """Compute average of all values equal to or above threshold (if specified)."""
    if (threshold or 0) > 0:
        data: pd.DataFrame = _compute_average_yearly_topic_weights_above_threshold(dtw, threshold, target_name)
        yearly_weights = yearly_weights.drop(columns=target_name).join(data, how='left').fillna(0)
    return yearly_weights


def _add_yearly_corpus_document_count(yearly_weights: pd.DataFrame, document_index: pd.DataFrame) -> pd.DataFrame:
    """Add a column with _total_ number of documents in corpus for given year."""
    yearly_weights: pd.DataFrame = yearly_weights.join(
        document_index.groupby('year').size().rename('n_documents'), how='left'
    ).fillna(0)
    return yearly_weights


def _add_average_yearly_topic_weight_by_all_documents(yearly_weights: pd.DataFrame) -> pd.DataFrame:
    """Compute "true" average weights (weight divided by total number of documents)"""
    yearly_weights['true_average_weight'] = yearly_weights.apply(lambda x: x['sum_weight'] / x['n_documents'], axis=1)
    return yearly_weights


def _add_top_n_topic_prevelance_weight(
    yearly_weights: pd.DataFrame, dtw: pd.DataFrame, n_top_relevance: int = None
) -> pd.DataFrame:

    if not n_top_relevance:
        return yearly_weights

    top_n_topics: pd.DataFrame = dtw.groupby(['year', 'document_id']).apply(
        lambda grp: grp.nlargest(n_top_relevance, 'weight')
    )['topic_id']
    top_n_counts: pd.DataFrame = (
        top_n_topics.reset_index().groupby(['year', 'topic_id'])['document_id'].size().rename('top_n_documents')
    )

    yearly_weights = yearly_weights.join(top_n_counts, how='left').fillna(0)
    yearly_weights['top_n_weight'] = yearly_weights.apply(lambda x: x['top_n_documents'] / x['n_documents'], axis=1)
    return yearly_weights


def compute_yearly_topic_weights(
    dtw: pd.DataFrame,
    *,
    document_index: pd.DataFrame,
    threshold: float = None,
    n_top_relevance: int = None,
    topic_ids: None | int | list[int] = None,
) -> pd.DataFrame:
    """Compute yearly document topic weights
         average_weight: average weight of all documents that has a weight in (i.e. assigned by engine) or a weight above a given threshold
    true_average_weight: average weight of all documents based on all documents (even documents where topic weight is 0)

    MALLET: Computes a weight for all topics in all documents
    GENSIM: Excludes topics having a weight less than 0
    """

    yearly_weights: pd.DataFrame = (
        _compute_yearly_topic_weights_statistics(dtw, topic_ids)
        .pipe(_add_yearly_corpus_document_count, document_index)
        .pipe(_add_average_yearly_topic_weight_by_all_documents)
        .pipe(_add_average_yearly_topic_weight_above_threshold, dtw, threshold, 'average_weight')
        .pipe(_add_top_n_topic_prevelance_weight, dtw, n_top_relevance)
        .reset_index()
    )
    # yearly_weights: pd.DataFrame = _compute_yearly_topic_weights_statistics(dtw, topic_ids)
    # yearly_weights = _add_yearly_corpus_document_count(yearly_weights, document_index)
    # yearly_weights = _add_average_yearly_topic_weight_by_all_documents(yearly_weights)
    # yearly_weights = _add_average_yearly_topic_weight_above_threshold(yearly_weights, dtw, threshold, 'average_weight')
    # yearly_weights = _add_top_n_topic_prevelance_weight(yearly_weights, dtw, n_top_relevance)
    # yearly_weights = yearly_weights.reset_index()

    return yearly_weights


AggregateKeys = Literal['top_n_weight', 'average_weight', 'true_average_weight']
AggregateKeyValues = get_args(AggregateKeys)

# @not_used
def compute_specified_yearly_topic_weights(
    dtw: pd.DataFrame,
    *,
    document_index: pd.DataFrame,
    aggregate_keys: str | AggregateKeys | list[AggregateKeys] = AggregateKeyValues,
    threshold: float = None,
    n_top_relevance: int = None,
    drop_stats: bool = True,
) -> pd.DataFrame:
    """Compute specified yearly document topic weight only"""
    fxs: dict = {
        'true_average_weight': (_add_average_yearly_topic_weight_by_all_documents, tuple()),
        'average_weight': (_add_average_yearly_topic_weight_above_threshold, (dtw, threshold, 'average_weight')),
        'top_n_weight': (
            _add_top_n_topic_prevelance_weight,
            (
                dtw,
                n_top_relevance,
            ),
        ),
    }
    aggregate_keys: list[str] = (
        aggregate_keys
        if isinstance(aggregate_keys, (list, tuple))
        else aggregate_keys.split(',')
        if isinstance(aggregate_keys, str)
        else AggregateKeyValues
    )

    ytw: pd.DataFrame = _compute_yearly_topic_weights_statistics(dtw).pipe(
        _add_yearly_corpus_document_count, document_index
    )

    if drop_stats:
        ytw.drop(columns=['max_weight', 'sum_weight', 'avg_weight'], inplace=True)

    for key in aggregate_keys:
        fx, args = fxs[key]
        ytw = fx(ytw, *args)

    return ytw.reset_index()
