import glob
import itertools
import os
import types
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd
import penelope.utility as utility
from penelope.topic_modelling.interfaces import ITopicModelEngine

from . import engine_gensim, engine_textacy

ENGINES = {'sklearn': engine_textacy, 'gensim_': engine_gensim}


def get_engine_module_by_method_name(method: str) -> types.ModuleType:
    for key in ENGINES:
        if method.startswith(key):
            return ENGINES[key]
    raise ValueError(f"Unknown method {method}")


def get_engine_cls_by_method_name(method: str) -> Type[ITopicModelEngine]:
    return get_engine_module_by_method_name(method).TopicModelEngine


def get_engine_by_model_type(model: Any) -> ITopicModelEngine:

    if engine_gensim.is_supported(model):
        return engine_gensim.TopicModelEngine(model)

    if engine_textacy.is_supported(model):
        return engine_textacy.TopicModelEngine(model)

    raise ValueError(f"unsupported model {type(model)}")


def find_models(path: str):
    """Return subfolders containing a computed topic model in specified path"""
    folders = [os.path.split(x)[0] for x in glob.glob(os.path.join(path, "*", "model_options.json"))]
    models = [
        {'folder': x, 'name': os.path.split(x)[1], 'options': utility.read_json(os.path.join(x, "model_options.json"))}
        for x in folders
    ]
    return models


# @deprecated
# def display_termite_plot(model, id2term, doc_term_matrix):
#     if hasattr(model, 'termite_plot'):
#         model.termite_plot(
#             doc_term_matrix,
#             id2term,
#             topics=-1,
#             sort_topics_by='index',
#             highlight_topics=None,
#             n_terms=50,
#             rank_terms_by='topic_weight',
#             sort_terms_by='seriation',
#             save=False,
#         )


def compute_topic_yearly_means(document_topic_weight: pd.DataFrame) -> pd.DataFrame:
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


def get_topic_titles(topic_token_weights: pd.DataFrame, topic_id: int = None, n_tokens: int = 100) -> pd.DataFrame:
    """Returns a DataFrame containing a string of `n_tokens` most probable words per topic"""

    weights: pd.DataFrame = (
        topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
    )

    df = (
        weights.sort_values('weight', ascending=False)
        .groupby('topic_id')
        .apply(lambda x: ' '.join(x.token[:n_tokens].str.title()))
    )

    return df


def get_topic_title(topic_token_weights: pd.DataFrame, topic_id: int, n_tokens: int = 100) -> str:
    """Returns a string of `n_tokens` most probable words per topic"""
    return get_topic_titles(topic_token_weights, topic_id, n_tokens=n_tokens).iloc[0]


def get_topic_top_tokens(topic_token_weights: pd.DataFrame, topic_id: int = None, n_tokens: int = 100) -> pd.DataFrame:
    """Returns most probable tokens for given topic sorted by probability descending"""
    weights = (
        topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
    )
    df = weights.sort_values('weight', ascending=False)[:n_tokens]
    return df


def get_topics_unstacked(
    model, n_tokens: int = 20, id2term: Dict[int, str] = None, topic_ids: List[int] = None
) -> pd.DataFrame:
    """Returns the top `n_tokens` tokens for each topic. The token's column index is in ascending probability"""

    engine: types.ModuleType = get_engine_by_model_type(model)
    n_topics = engine.n_topics()

    topic_ids = topic_ids or range(n_topics)

    return pd.DataFrame(
        {
            'Topic#{:02d}'.format(topic_id + 1): [
                word[0]
                for word in engine.topic_tokens(model=model, topic_id=topic_id, n_tokens=n_tokens, id2term=id2term)
            ]
            for topic_id in topic_ids
        }
    )
