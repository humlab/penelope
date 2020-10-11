import json
import os
import pickle

import penelope.utility as utility

from . import engine_gensim, engine_textacy
from .container import InferredModel, TrainingCorpus
from .utility import add_document_terms_count

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

    train_corpus.documents = add_document_terms_count(train_corpus.documents, train_corpus.corpus)

    return inferred_model


def store_model(inferred_model: InferredModel, folder: str, store_corpus: bool = False):
    """Stores an inferred model in icled format. Train corpus is not stored."""

    os.makedirs(folder, exist_ok=True)

    if not store_corpus:
        inferred_model.train_corpus = None

    filename = os.path.join(folder, "inferred_model.pickle")

    with open(filename, 'wb') as fp:
        pickle.dump(inferred_model, fp, pickle.HIGHEST_PROTOCOL)

    filename = os.path.join(folder, "model_options.json")
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    with open(filename, 'w') as fp:
        json.dump(inferred_model.options, fp, indent=4, default=default)


def load_model(folder: str) -> InferredModel:
    """Loads an inferred model from av previously pickled file."""
    filename = os.path.join(folder, "inferred_model.pickle")
    with open(filename, 'rb') as f:
        inferred_model = pickle.load(f)
    return inferred_model
