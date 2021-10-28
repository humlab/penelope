import glob
import itertools
import os
from typing import Any, List, Mapping, Set

import numpy as np
import pandas as pd
import penelope.utility as utility
import scipy.sparse as sp
from loguru import logger
from penelope.type_alias import DocumentIndex


def find_models(path: str) -> dict:
    """Return subfolders containing a computed topic model in specified path"""
    folders = [os.path.split(x)[0] for x in glob.glob(os.path.join(path, "*", "model_options.json"))]
    models = [
        {'folder': x, 'name': os.path.split(x)[1], 'options': utility.read_json(os.path.join(x, "model_options.json"))}
        for x in folders
    ]
    return models


def find_inferred_topics_folders(folder: str) -> List[str]:
    """Return inferred data in sub-folders to `folder`"""
    filenames = glob.glob(os.path.join(folder, "**/*document_topic_weights.zip"), recursive=True)
    folders = [os.path.split(filename)[0] for filename in filenames]
    return folders


def compute_topic_yearly_means(
    document_topic_weight: pd.DataFrame, relevence_mean_threshold: float = None
) -> pd.DataFrame:
    """Returs yearly mean topic weight based on data in `document_topic_weight`"""

    cross_iter = itertools.product(
        range(document_topic_weight.year.min(), document_topic_weight.year.max() + 1),
        range(0, document_topic_weight.topic_id.max() + 1),
    )
    dfs = pd.DataFrame(list(cross_iter), columns=['year', 'topic_id']).set_index(['year', 'topic_id'])

    """ Add the most basic stats """
    dfs = dfs.join(
        document_topic_weight.groupby(['year', 'topic_id'])['weight'].agg([np.max, np.sum, np.mean, len]), how='left'
    ).fillna(0)

    dfs.columns = ['max_weight', 'sum_weight', 'false_mean', 'n_topic_docs']

    dfs['n_topic_docs'] = dfs.n_topic_docs.astype(np.uint32)

    if relevence_mean_threshold is not None:

        dfs.drop(columns='false_mean', inplace=True)

        df_mean_relevance = (
            document_topic_weight[document_topic_weight.weight >= relevence_mean_threshold]
            .groupby(['year', 'topic_id'])['weight']
            .agg([np.mean])
        )
        df_mean_relevance.columns = ['false_mean']

        dfs = dfs.join(df_mean_relevance, how='left').fillna(0)

    doc_counts = document_topic_weight.groupby('year').document_id.nunique().rename('n_total_docs')

    dfs = dfs.join(doc_counts, how='left').fillna(0)
    dfs['n_total_docs'] = dfs.n_total_docs.astype(np.uint32)
    dfs['true_mean'] = dfs.apply(lambda x: x['sum_weight'] / x['n_total_docs'], axis=1)

    return dfs.reset_index()


# @deprecated
# def normalize_weights(df: pd.DataFrame):

#     dfy = df.groupby(['year'])['weight'].sum().rename('sum_weight')
#     df = df.merge(dfy, how='inner', left_on=['year'], right_index=True)
#     df['weight'] = df.apply(lambda x: x['weight'] / x['sum_weight'], axis=1)
#     df = df.drop(['sum_weight'], axis=1)
#     return df


def get_topic_titles(topic_token_weights: pd.DataFrame, topic_id: int = None, n_tokens: int = 100) -> pd.Series:
    """Create string of `n_tokens` most probable words per topic."""

    weights: pd.DataFrame = (
        topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
    )

    topic_titles: pd.DataFrame = (
        weights.sort_values('weight', ascending=False)
        .groupby('topic_id')
        .apply(lambda x: ' '.join(x.token[:n_tokens].str.title()))
    )

    return topic_titles


def get_topic_title(topic_token_weights: pd.DataFrame, topic_id: int, n_tokens: int = 100) -> str:
    """Returns a string of `n_tokens` most probable words for topic `topic_id`"""
    return get_topic_titles(topic_token_weights, topic_id, n_tokens=n_tokens).iloc[0]


def get_topic_title2(topic_token_weights: pd.DataFrame, topic_id: int, n_tokens: int = 200) -> str:
    """Returns a string of `n_tokens` most probable words for topic `topic_id` or message if not tokens."""
    if len(topic_token_weights[topic_token_weights.topic_id == topic_id]) == 0:
        tokens = "Topics has no significant presence in any documents in the entire corpus"
    else:
        tokens = get_topic_title(topic_token_weights, topic_id, n_tokens=n_tokens)

    return f'ID {topic_id}: {tokens}'


def get_topic_top_tokens(topic_token_weights: pd.DataFrame, topic_id: int = None, n_tokens: int = 100) -> pd.DataFrame:
    """Returns most probable tokens for given topic sorted by probability descending"""
    weights = (
        topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
    )
    df = weights.sort_values('weight', ascending=False)[:n_tokens]
    return df


def top_topic_token_weights(topic_token_weights: pd.DataFrame, id2term: dict, n_count: int) -> pd.DataFrame:
    """Find top `n_count` tokens for each topic. Return data frame."""
    _largest = (
        topic_token_weights.groupby(['topic_id'])[['topic_id', 'token_id', 'weight']]
        .apply(lambda x: x.nlargest(n_count, columns=['weight']))
        .reset_index(drop=True)
    )
    _largest['token'] = _largest.token_id.apply(lambda x: id2term[x])
    _largest['position'] = _largest.groupby('topic_id').cumcount() + 1
    return _largest.set_index('topic_id')


def top_topic_token_weights_old(topic_token_weights: pd.DataFrame, id2term: dict, n_count: int) -> pd.DataFrame:
    _largest = topic_token_weights.groupby(['topic_id'])[['topic_id', 'token_id', 'weight']].apply(
        lambda x: x.nlargest(n_count, columns=['weight'])
    )
    _largest['token'] = _largest.token_id.apply(lambda x: id2term[x])
    return _largest.set_index('topic_id')


def _compute_topic_proportions(document_topic_weights: pd.DataFrame, doc_length_series: np.ndarray) -> np.ndarray:
    """Compute topic proportations the LDAvis way. Fast version"""
    theta: sp.coo_matrix = sp.coo_matrix(
        (document_topic_weights.weight, (document_topic_weights.document_id, document_topic_weights.topic_id))
    )
    theta_mult_doc_length: np.ndarray = theta.T.multiply(doc_length_series).T
    topic_frequency: np.ndarray = theta_mult_doc_length.sum(axis=0).A1
    topic_proportion: np.ndarray = topic_frequency / topic_frequency.sum()
    return topic_proportion


def compute_topic_proportions(document_topic_weights: pd.DataFrame, document_index: pd.DataFrame) -> pd.DataFrame:
    """Compute topics' proportion in entire corpus."""
    if 'n_terms' not in document_index.columns:
        return None

    return _compute_topic_proportions(document_topic_weights, document_index.n_terms.values)


def filter_document_topic_weights(
    document_topic_weights: pd.DataFrame, filters: Mapping[str, Any] = None, threshold: float = 0.0
) -> pd.DataFrame:
    """Returns document's topic weights for given `year`, `topic_id`, custom `filters` and threshold.

    Parameters
    ----------
    document_topic_weights : pd.DataFrame
        Document topic weights
    filters : Dict[str, Any], optional
        [description], by default None
    threshold : float, optional
        [description], by default 0.0

    Returns
    -------
    pd.DataFrame
        [description]
    """
    df: pd.DataFrame = document_topic_weights

    df = df[df.weight >= threshold]

    for k, v in (filters or {}).items():
        if k not in df.columns:
            logger.warning('Column %s does not exist in dataframe (_find_documents_for_topics)', k)
            continue
        df = df[df[k] == v]

    return df.copy()


def get_topic_documents(
    document_topic_weights: pd.DataFrame,
    document_index: DocumentIndex,
    threshold: float = 0.0,
    n_top: int = 500,
    **filters,
) -> pd.DataFrame:
    topic_documents = filter_document_topic_weights(document_topic_weights, filters=filters, threshold=threshold)
    if len(topic_documents) == 0:
        return None

    topic_documents = (
        topic_documents.drop(['topic_id'], axis=1)
        .set_index('document_id')
        .sort_values('weight', ascending=False)
        .head(n_top)
    )
    additional_columns = [x for x in document_index.columns.tolist() if x not in ['year', 'document_name']]
    topic_documents = topic_documents.merge(
        document_index[additional_columns], left_index=True, right_on='document_id', how='inner'
    )
    topic_documents.index.name = 'id'
    return topic_documents


def filter_topic_tokens_overview(
    topic_tokens_overview: pd.DataFrame, *, search_text: str, n_count: int
) -> pd.DataFrame:
    """Filter out topics where `search` string is in `n_counts` words. Return data frame."""
    reduced_topics = (
        topic_tokens_overview[topic_tokens_overview.tokens.str.contains(search_text)]
        if search_text
        else topic_tokens_overview
    )

    tokens: pd.Series = reduced_topics.tokens.apply(lambda x: " ".join(x.split()[:n_count]))

    if search_text:
        tokens = reduced_topics.tokens.apply(
            lambda x: x.replace(search_text, f'<b style="color:green;font-size:14px">{search_text}</b>')
        )

    reduced_topics['tokens'] = tokens

    return reduced_topics


class DocumentTopicWeights:
    def __init__(self, document_topic_weights: pd.DataFrame, document_index: pd.DataFrame):

        self.document_topic_weights: pd.DataFrame = document_topic_weights
        self.document_index: pd.DataFrame = document_index
        self.data: pd.DataFrame = document_topic_weights

    def filter_by(
        self,
        threshold: float = 0.0,
        key_values: Mapping[str, Any] = None,
        document_key_values: Mapping[str, Any] = None,
    ) -> "DocumentTopicWeights":
        return self.threshold(threshold).filter_by_keys(key_values).filter_by_document_keys(document_key_values)

    def threshold(self, threshold: float = 0.0) -> "DocumentTopicWeights":
        """Filter document-topic weights by threshold"""

        if threshold > 0:
            self.data = self.data[self.data.weight >= threshold]

        return self

    @property
    def copy(self) -> pd.DataFrame:
        return self.data.copy()

    @property
    def value(self) -> pd.DataFrame:
        return self.data

    def filter_by_keys(self, key_values: Mapping[str, Any] = None) -> "DocumentTopicWeights":
        """Filter data by key values. Returnm self."""
        if key_values is not None:
            self.data = self.data[utility.create_mask(self.data, key_values)]
        return self

    def filter_by_document_keys(self, key_values: Mapping[str, Any] = None) -> "DocumentTopicWeights":
        """Filter data by key values. Returnm self."""

        if key_values is not None:

            mask: np.ndarray = utility.create_mask(self.document_index, key_values)

            document_index: pd.DataFrame = self.document_index[mask]
            document_ids: Set[int] = set(document_index.document_id.unique())

            self.data = self.data[self.data.document_id.isin(document_ids)]

        return self
