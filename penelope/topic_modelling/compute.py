import bz2
import json
import os
import pickle
from enum import Enum
from typing import Any, Dict

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


class StoreCorpusOptions(Enum):
    NONE = 1
    PICKLED = 2


def _compressed_pickle(filename: str, thing: Any):
    with bz2.BZ2File(filename, 'w') as f:
        pickle.dump(thing, f)


def _compressed_unpickle(filename):
    with bz2.BZ2File(filename, 'rb') as f:
        data = pickle.load(f)
        return data


def _pickle(filename: str, thing: Any):
    """Pickles a thing to disk """
    if filename.endswith('.pbz2'):
        _compressed_pickle(filename, thing)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(thing, f, pickle.HIGHEST_PROTOCOL)


def _unpickle(filename: str) -> Any:
    """Unpickles a thing from disk."""
    if filename.endswith('.pbz2'):
        thing = _compressed_unpickle(filename)
    else:
        with open(filename, 'rb') as f:
            thing = pickle.load(f)
    return thing


def _store_train_corpus(folder: str, train_corpus: TrainingCorpus, store_compressed: bool = True):

    """Stores the corpus used in training. If not pickled, then stored as separate files
    terms: Iterable[Iterable[str]]                               Never stored
    documents: pd.DataFrame                                      TODO Stored as csv.zip
    doc_term_matrix: scipy.sparse.csr_matrix                     Never stored
    id2word: Union[gensim.corpora.Dictionary, Dict[int, str]]    TODO Stored compressed as gensim.Dictionar
    vectorizer_args: Dict[str, Any]                              TODO Stored as json
    corpus: ???                                                  Stored as SparseCorpus
    """
    filename = os.path.join(folder, f"training_corpus.pickle{'.pbz2' if store_compressed else ''}")

    _train_corpus = TrainingCorpus(
        doc_term_matrix=None,
        terms=None,
        corpus=train_corpus.corpus,
        documents=train_corpus.documents,
        id2word=train_corpus.id2word,
        vectorizer_args=train_corpus.vectorizer_args,
    )
    _pickle(filename, _train_corpus)


def _load_train_corpus(folder: str) -> TrainingCorpus:
    """Loads an train corpus from av previously pickled file."""
    filename = os.path.join(folder, "training_corpus.pickle.pbz2")
    if not os.path.isfile(filename):
        return None
    return _unpickle(os.path.join(folder, "training_corpus.pickle.pbz2"))


def _store_topic_model(folder: str, topic_model: Any, store_compressed: bool = True):
    """Stores topic model in pickled format """
    filename = os.path.join(folder, f"topic_model.pickle{'.pbz2' if store_compressed else ''}")
    _pickle(filename, topic_model)


def _load_topic_model(folder: str) -> Any:
    """Loads an train corpus from av previously pickled file."""
    return _unpickle(os.path.join(folder, "topic_model.pickle.pbz2"))


def _store_model_options(folder: str, method: str, options: Dict[str, Any]):
    filename = os.path.join(folder, "model_options.json")
    options = {**dict(method=method), **options}
    with open(filename, 'w') as fp:
        json.dump(options, fp, indent=4, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")


def _load_model_options(folder: str) -> Dict[str, Any]:
    filename = os.path.join(folder, "model_options.json")
    with open(filename, 'r') as f:
        options = json.load(f)
    return options


def store_model(inferred_model: InferredModel, folder: str, store_corpus=True, store_compressed=True):
    """Stores the inferred model on disk in folder `folder`

        topic_model: Gensim | MALLET | STTM     Stored in 'topic_model.pickle'
        train_corpus: TrainingCorpus            Stored in
        method: str                             Stored in JSON infer.options.json: method
        options: Dict[str, Any]                 Stored in JSON infer.options.json: engine_args

    Parameters
    ----------
    inferred_model : InferredModel
        Model to be stored
    folder : str
        Target folder
    store_corpus : bool, optional
        Specifies if training corpus should be stored, by default False
    """

    os.makedirs(folder, exist_ok=True)

    _store_topic_model(folder, inferred_model.topic_model, store_compressed=store_compressed)

    if store_corpus:
        _store_train_corpus(folder, inferred_model.train_corpus, store_compressed=store_compressed)

    _store_model_options(folder, method=inferred_model.method, options=inferred_model.options)


def load_model(folder: str) -> InferredModel:
    """Loads inferred model data from previously pickled files."""
    topic_model = _load_topic_model(folder)
    train_corpus = _load_train_corpus(folder)
    options = _load_model_options(folder)
    return InferredModel(topic_model=topic_model, train_corpus=train_corpus, **options)
