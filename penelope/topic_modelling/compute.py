import json
import os
import pickle
import types

import penelope.topic_modelling.engine_gensim as engine_gensim
import penelope.topic_modelling.engine_textacy as engine_textacy
import penelope.utility as utility
from penelope.topic_modelling.container import \
    ModelAgnosticDataContainer
from penelope.topic_modelling.extract import (extract_topic_token_overview,
                                              extract_topic_token_weights)

from .predict import predict_document_topics
from .utility import document_n_terms, id2word2dataframe

logger = utility.getLogger("")

TEMP_PATH = './tmp/'

# OBS OBS! https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)

engines = {'sklearn': engine_textacy, 'gensim_': engine_gensim}


def _find_engine(method):
    for key in engines:
        if method.startswith(key):
            return engines[key]
    raise ValueError(f"Unknown method {method}")


# class ComputeCorpus:

#     def __init__(self, terms=None, documents=None, doc_term_matrix=None, id2word=None):
#         self.terms = terms
#         self.doc_term_matrix = doc_term_matrix
#         self.id2word = id2word
#         self.documents = documents


def compute_model(
    terms=None,
    documents=None,
    doc_term_matrix=None,
    id2word=None,
    method: str = 'sklearn_lda',
    vectorizer_args=None,
    engine_args=None,
    **args,
):

    vectorizer_args = {**DEFAULT_VECTORIZE_PARAMS, **(vectorizer_args or {})}

    os.makedirs(TEMP_PATH, exist_ok=True)

    engine = _find_engine(method)

    result = engine.compute(
        doc_term_matrix,
        terms,
        id2word,
        vectorizer_args,
        method,
        engine_args,
        tfidf_weiging=args.get('tfidf_weiging', False)
    )

    """ Fix missing n_terms (only vectorized corps"""
    if 'n_terms' not in documents.columns:
        n_terms = document_n_terms(result.corpus)
        if n_terms is not None:
            documents['n_terms'] = n_terms

    # FIXME: Seperate model data from generated and predicted data
    # Generate model agnostic data
    dictionary = id2word2dataframe(id2word)
    topic_token_weights = extract_topic_token_weights(result.model, dictionary, n_tokens=200)
    topic_token_overview = extract_topic_token_overview(result.model, topic_token_weights)

    document_topic_weights = predict_document_topics(
        result.model,
        result.corpus,
        documents=documents,
        doc_topic_matrix=result.doc_topic_matrix,
        minimum_probability=0.001
    )

    c_data = ModelAgnosticDataContainer(documents, dictionary, topic_token_weights, topic_token_overview, document_topic_weights)

    m_data = types.SimpleNamespace(
        topic_model=result.model,
        id2term=id2word,
        corpus=result.corpus,  # FIXME: Remove (train) corpus from model data
        metrics=dict(coherence_score=result.coherence_score, perplexity_score=result.perplexity_score),
        options=dict(
            method=method,
            vec_args=vectorizer_args,
            tm_args=result.engine_options,
            **args,
        ),
        coherence_scores=None,
    )

    return m_data, c_data


def store_model(model_data, folder):

    os.makedirs(folder, exist_ok=True)

    model_data.doc_term_matrix = None
    model_data.options['tm_args']['id2word'] = None
    model_data.options['tm_args']['corpus'] = None

    filename = os.path.join(folder, "model_data.pickle")

    with open(filename, 'wb') as fp:
        pickle.dump(model_data, fp, pickle.HIGHEST_PROTOCOL)

    filename = os.path.join(folder, "model_options.json")
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    with open(filename, 'w') as fp:
        json.dump(model_data.options, fp, indent=4, default=default)


def load_model(folder):
    filename = os.path.join(folder, "model_data.pickle")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
