import json
import os
import pickle

import penelope.utility as utility

from . import engine_gensim, engine_textacy
from .container import InferredModel, InferredTopicsData, TrainingCorpus
from .extract import extract_topic_token_overview, extract_topic_token_weights
from .predict import predict_document_topics
from .utility import id2word_to_dataframe

logger = utility.getLogger("")

TEMP_PATH = './tmp/'

# OBS OBS! https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

engines = {'sklearn': engine_textacy, 'gensim_': engine_gensim}


def _find_engine(method):
    for key in engines:
        if method.startswith(key):
            return engines[key]
    raise ValueError(f"Unknown method {method}")


def infer_model(
    train_corpus: TrainingCorpus,
    method: str = 'sklearn_lda',
    engine_args=None,
    **kwargs,
) -> InferredModel:

    os.makedirs(TEMP_PATH, exist_ok=True)

    inferred_model = _find_engine(method).compute(
        train_corpus,
        method,
        engine_args,
        tfidf_weiging=kwargs.get('tfidf_weiging', False),
    )

    # Generate model agnostic data
    n_tokens = kwargs.get('n_tokens', 200)
    dictionary = id2word_to_dataframe(train_corpus.id2word)
    topic_token_weights = extract_topic_token_weights(inferred_model.topic_model, dictionary, n_tokens=n_tokens)
    topic_token_overview = extract_topic_token_overview(
        inferred_model.topic_model, topic_token_weights, n_tokens=n_tokens
    )

    # """ Fix missing n_terms (only vectorized corps"""
    # if 'n_terms' not in train_corpus.documents.columns:
    #     n_terms = document_n_terms(inferred_model.corpus)
    #     if n_terms is not None:
    #         train_corpus.documents['n_terms'] = n_terms

    document_topic_weights = predict_document_topics(
        inferred_model.topic_model,
        train_corpus.corpus,
        documents=train_corpus.documents,
        minimum_probability=0.001,
    )

    inferred_topics_data = InferredTopicsData(
        train_corpus.documents, dictionary, topic_token_weights, topic_token_overview, document_topic_weights
    )

    return inferred_model, inferred_topics_data


def store_model(inferred_model: InferredModel, folder: str):
    """Stores an inferred model in icled format. Train corpus is not stored."""
    os.makedirs(folder, exist_ok=True)

    inferred_model.train_corpus.doc_term_matrix = None
    inferred_model.train_corpus.id2word = None
    inferred_model.train_corpus.corpus = None
    inferred_model.train_corpus.terms = None

    filename = os.path.join(folder, "model_data.pickle")

    with open(filename, 'wb') as fp:
        pickle.dump(inferred_model, fp, pickle.HIGHEST_PROTOCOL)

    filename = os.path.join(folder, "model_options.json")
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    with open(filename, 'w') as fp:
        json.dump(inferred_model.options, fp, indent=4, default=default)


def load_model(folder: str) -> InferredModel:
    """Loads an inferred model from av previously pickled file."""
    filename = os.path.join(folder, "model_data.pickle")
    with open(filename, 'rb') as f:
        inferred_model = pickle.load(f)
    return inferred_model
