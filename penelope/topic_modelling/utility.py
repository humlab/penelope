import glob
import itertools
import os

import gensim
import numpy as np
import pandas as pd
import scipy

import penelope.utility as utility


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


def malletmodel2ldamodel(
        mallet_model:gensim.models.wrappers.ldamallet.LdaMallet,
        gamma_threshold: float=0.001,
        iterations: int=50
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
        dtype=np.float64  # don't loose precision when converting from MALLET
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim


def find_models(path: str):
    """Returns subfolders containing a computed topic model in specified path"""
    folders = [os.path.split(x)[0] for x in glob.glob(os.path.join(path, "*", "model_data.pickle"))]
    models = [{
        'folder': x,
        'name': os.path.split(x)[1],
        'options': utility.read_json(os.path.join(x, "model_options.json"))
    } for x in folders]
    return models

def display_termite_plot(model, id2term, doc_term_matrix):

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
            save=False
        )


METHODS = [{
    'key': 'max_weight',
    'description': 'Max value',
    'tooltip': 'Use maximum value over documents'
}, {
    'key': 'false_mean',
    'description': 'Mean where topic is relevant',
    'tooltip': 'Use mean value of all documents where topic is above certain treshold'
}, {
    'key': 'true_mean',
    'description': 'Mean of all documents',
    'tooltip': 'Use mean value of all documents even those where topic is zero'
}]


def plot_topic(df, x):
    df = df.reset_index()
    df[df.topic_id == x].set_index('year').drop('topic_id', axis=1).plot()


def compute_means(df):
    """ Initialize year/topic cross product data frame """
    cross_iter = itertools.product(range(df.year.min(), df.year.max() + 1), range(0, df.topic_id.max() + 1))
    dfs = pd.DataFrame(list(cross_iter), columns=['year', 'topic_id']).set_index(['year', 'topic_id'])
    """ Add the most basic stats """
    dfs = dfs.join(df.groupby(['year', 'topic_id'])['weight'].agg([np.max, np.sum, np.mean, len]), how='left').fillna(0)
    dfs.columns = ['max_weight', 'sum_weight', 'false_mean', 'n_topic_docs']
    dfs['n_topic_docs'] = dfs.n_topic_docs.astype(np.uint32)

    doc_counts = df.groupby('year').document_id.nunique().rename('n_total_docs')
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
