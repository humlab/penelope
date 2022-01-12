import glob
import os
from typing import Any, List, Mapping

import numpy as np
import pandas as pd
import penelope.utility as utility
import scipy.sparse as sp
from loguru import logger
from penelope.type_alias import DocumentIndex
from penelope.utility import deprecated


def find_models(path: str) -> dict:
    """Return subfolders containing a computed topic model in specified path"""
    folders = [os.path.split(x)[0] for x in glob.glob(os.path.join(path, "**", "model_options.json"), recursive=True)]
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


def top_topic_token_weights(topic_token_weights: pd.DataFrame, id2term: dict, n_top: int) -> pd.DataFrame:
    """Find top `n_top` tokens for each topic. Return data frame."""
    _largest = (
        topic_token_weights.groupby(['topic_id'])[['topic_id', 'token_id', 'weight']]
        .apply(lambda x: x.nlargest(n_top, columns=['weight']))
        .reset_index(drop=True)
    )
    _largest['token'] = _largest.token_id.apply(lambda x: id2term[x])
    _largest['position'] = _largest.groupby('topic_id').cumcount() + 1
    return _largest.set_index('topic_id')


@deprecated
def top_topic_token_weights_old(topic_token_weights: pd.DataFrame, id2term: dict, n_top: int) -> pd.DataFrame:
    _largest = topic_token_weights.groupby(['topic_id'])[['topic_id', 'token_id', 'weight']].apply(
        lambda x: x.nlargest(n_top, columns=['weight'])
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
    if 'n_tokens' not in document_index.columns:
        return None

    return _compute_topic_proportions(document_topic_weights, document_index.n_tokens.values)


# FIXME: Deprecate method. Use DocumentTopicWeights instead
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


def get_relevant_topic_documents(
    document_topic_weights: pd.DataFrame,
    document_index: DocumentIndex,
    threshold: float = 0.0,
    n_top: int = 500,
    **filters,
) -> pd.DataFrame:
    """Generate list of documents where topics are deemed relevant"""
    topic_documents = filter_document_topic_weights(document_topic_weights, filters=filters, threshold=threshold)
    if len(topic_documents) == 0:
        return None

    topic_documents = (
        topic_documents.set_index('document_id')  # .drop(['topic_id'], axis=1)
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
    topic_tokens_overview: pd.DataFrame,
    *,
    search_text: str,
    n_top: int,
    truncate_tokens: bool = False,
    format_string: str = '<b style="color:green;font-size:14px">{}</b>',
) -> pd.DataFrame:
    """Filter out topics where `search` string is in `n_counts` words. Return data frame."""

    data = pd.DataFrame(topic_tokens_overview)

    if search_text:
        top_tokens = data.tokens.apply(lambda x: x.split(' ')[:n_top]).str.join(' ')
        data = data[top_tokens.str.contains(search_text)]
        data['tokens'] = (top_tokens if truncate_tokens else data.tokens).apply(
            lambda x: x.replace(search_text, format_string.format(search_text))
        )

    return data
