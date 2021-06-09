import glob
import itertools
import os
from typing import Dict, List

import gensim
import numpy as np
import pandas as pd
import penelope.utility as utility
import scipy

logger = utility.getLogger('corpus_text_analysis')

# FIXME: #99 Add type hints
def compute_topic_proportions(document_topic_weights: pd.DataFrame, doc_length_series: np.ndarray):
    """Computes topic proportations as LDAvis. Fast version
    Parameters
    ----------
    document_topic_weights : :class:`~pandas.DataFrame`
        Document Topic Weights
    doc_length_series : numpy.ndarray
        Document lengths
    Returns
    -------
    numpy array
    """
    theta = scipy.sparse.coo_matrix(
        (document_topic_weights.weight, (document_topic_weights.document_id, document_topic_weights.topic_id))
    )
    theta_mult_doc_length = theta.T.multiply(doc_length_series).T
    topic_frequency = theta_mult_doc_length.sum(axis=0).A1
    topic_proportion = topic_frequency / topic_frequency.sum()
    return topic_proportion


# def compute_topic_proportions2(document_topic_weights: pd.DataFrame) -> pd.DataFrame:
#     """Computes topic proportions (used by InferedTopicsData)"""

#     if 'n_raw_tokens' not in document_topic_weights.columns:
#         logger.info("warning: unable to compute topic proportions ('n_raw_tokens' not found in document_topic_weights)")
#         return None

#     doc_topic_dists: pd.DataFrame = document_topic_weights[['document_id', 'topic_id', 'weight', 'n_raw_tokens']]
#     # compute sum of (topic weight x document lengths)
#     topic_freqs: pd.DataFrame = (
#         doc_topic_dists.assign(t_weight=lambda df: df.weight * df.n_raw_tokens).groupby('topic_id')['t_weight'].sum()
#     )
#     # normalize on total sum
#     topic_proportion: pd.Series = (topic_freqs / topic_freqs.sum()).sort_values(ascending=False)
#     # return global topic proportion
#     topic_proportion = pd.DataFrame(data={'topic_proportion': 100.0 * topic_proportion})

#     return topic_proportion


# FIXME #98 gensim 4.0: wrappers.ldamallet.LdaMallet is deeprecated/removed in Gensim 4.0
def malletmodel2ldamodel(
    mallet_model: gensim.models.wrappers.ldamallet.LdaMallet, gamma_threshold: float = 0.001, iterations: int = 50
):
    """Convert :class:`~gensim.models.wrappers.ldamallet.LdaMallet` to :class:`~gensim.models.ldamodel.LdaModel`.
    This works by copying the training model weights (alpha, beta...) from a trained mallet model into the gensim model.
    Parameters
    ----------
    mallet_model : :class:`~gensim.models.wrappers.ldamallet.LdaMallet`
        Trained Mallet model
    gamma_threshold : float, optional
        To be used for inference in the new LdaModel.
    iterations : int, optional
        Number of iterations to be used for inference in the new LdaModel.
    Returns
    -------
    :class:`~gensim.models.ldamodel.LdaModel`
        Gensim native LDA.
    """
    model_gensim = gensim.models.LdaModel(
        id2word=mallet_model.id2word,
        num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha,
        eta=0,
        iterations=iterations,
        gamma_threshold=gamma_threshold,
        dtype=np.float64,  # don't loose precision when converting from MALLET
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim


def find_models(path: str):
    """Returns subfolders containing a computed topic model in specified path"""
    folders = [os.path.split(x)[0] for x in glob.glob(os.path.join(path, "*", "topic_model.pickle*"))]
    models = [
        {'folder': x, 'name': os.path.split(x)[1], 'options': utility.read_json(os.path.join(x, "model_options.json"))}
        for x in folders
    ]
    return models


def display_termite_plot(model, id2term, doc_term_matrix):
    """[summary]

    Parameters
    ----------
    model : [type]
        [description]
    id2term : [type]
        [description]
    doc_term_matrix : [type]
        [description]
    """
    if hasattr(model, 'termite_plot'):
        model.termite_plot(
            doc_term_matrix,
            id2term,
            topics=-1,
            sort_topics_by='index',
            highlight_topics=None,
            n_terms=50,
            rank_terms_by='topic_weight',
            sort_terms_by='seriation',
            save=False,
        )


YEARLY_MEAN_COMPUTE_METHODS = [
    {'key': 'max_weight', 'description': 'Max value', 'tooltip': 'Use maximum value over documents'},
    {
        'key': 'false_mean',
        'description': 'Mean where topic is relevant',
        'tooltip': 'Use mean value of all documents where topic is above certain treshold',
    },
    {
        'key': 'true_mean',
        'description': 'Mean of all documents',
        'tooltip': 'Use mean value of all documents even those where topic is zero',
    },
]


def plot_topic(df, x):
    df = df.reset_index()
    df[df.topic_id == x].set_index('year').drop('topic_id', axis=1).plot()


def compute_topic_yearly_means(document_topic_weight: pd.DataFrame) -> pd.DataFrame:
    """Returns yearly mean topic weight based on data in `document_topic_weight`

    Parameters
    ----------
    document_topic_weight : pd.DataFrame
        The DTM

    Returns
    ----------
    pd.DataFrame
        The yearly max, mean values for the topic weights, the latter is computed for all all documents.

    """
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


def normalize_weights(df):

    dfy = df.groupby(['year'])['weight'].sum().rename('sum_weight')
    df = df.merge(dfy, how='inner', left_on=['year'], right_index=True)
    df['weight'] = df.apply(lambda x: x['weight'] / x['sum_weight'], axis=1)
    df = df.drop(['sum_weight'], axis=1)
    return df


def document_terms_count(corpus):

    n_terms = None

    if hasattr(corpus, 'sparse'):
        n_terms = corpus.sparse.sum(axis=0).A1

    if isinstance(corpus, list):
        n_terms = [sum((w[1] for w in d)) for d in corpus]

    try:
        n_terms = [len(d) for d in corpus]
    except:  # pylint:disable=bare-except
        pass

    return n_terms


def add_document_terms_count(document_index, corpus):
    if 'n_terms' not in document_index.columns:
        n_terms = document_terms_count(corpus)
        if n_terms is not None:
            document_index['n_terms'] = n_terms
    return document_index


def id2word_to_dataframe(id2word: Dict) -> pd.DataFrame:
    """Returns token id to word mapping `id2word` as a pandas DataFrane, with DFS added

    Parameters
    ----------
    id2word : Dict
        Token ID to word mapping

    Returns
    -------
    pd.DataFrame
        dictionary as dataframe
    """
    logger.info('Compiling dictionary...')

    assert id2word is not None, 'id2word is empty'

    dfs = list(id2word.dfs.values()) or 0 if hasattr(id2word, 'dfs') else 0

    token_ids, tokens = list(zip(*id2word.items()))

    dictionary = pd.DataFrame({'token_id': token_ids, 'token': tokens, 'dfs': dfs}).set_index('token_id')[
        ['token', 'dfs']
    ]

    return dictionary


def get_topic_titles(topic_token_weights: pd.DataFrame, topic_id: int = None, n_tokens: int = 100) -> pd.DataFrame:
    """Returns a DataFrame containing a string of `n_tokens` most probable words per topic"""

    df_temp = (
        topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
    )

    df = (
        df_temp.sort_values('weight', ascending=False)
        .groupby('topic_id')
        .apply(lambda x: ' '.join(x.token[:n_tokens].str.title()))
    )

    return df


def get_topic_title(topic_token_weights: pd.DataFrame, topic_id: int, n_tokens: int = 100) -> str:
    """Returns a string of `n_tokens` most probable words per topic"""
    return get_topic_titles(topic_token_weights, topic_id, n_tokens=n_tokens).iloc[0]


def get_topic_tokens(topic_token_weights: pd.DataFrame, topic_id: int = None, n_tokens: int = 100) -> pd.DataFrame:
    """Returns most probable tokens for given topic sorted by probability descending"""
    df_temp = (
        topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
    )
    df = df_temp.sort_values('weight', ascending=False)[:n_tokens]
    return df


def get_topics_unstacked(
    model, n_tokens: int = 20, id2term: Dict[int, str] = None, topic_ids: List[int] = None
) -> pd.DataFrame:
    """Returns the top `n_tokens` tokens for each topic. The token's column index is in ascending probability"""

    if hasattr(model, 'num_topics'):
        # Gensim LDA model
        show_topic = lambda topic_id: model.show_topic(topic_id, topn=n_tokens)
        n_topics = model.num_topics
    elif hasattr(model, 'm_T'):
        # Gensim HDP model
        show_topic = lambda topic_id: model.show_topic(topic_id, topn=n_tokens)
        n_topics = model.m_T
    else:
        # Textacy/scikit-learn model
        def scikit_learn_show_topic(topic_id):
            topic_words = list(model.top_topic_terms(id2term, topics=(topic_id,), top_n=n_tokens, weights=True))
            if len(topic_words) == 0:
                return []
            return topic_words[0][1]

        show_topic = scikit_learn_show_topic
        n_topics = model.n_topics

    topic_ids = topic_ids or range(n_topics)

    return pd.DataFrame(
        {'Topic#{:02d}'.format(topic_id + 1): [word[0] for word in show_topic(topic_id)] for topic_id in topic_ids}
    )
