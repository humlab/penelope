import json
import os
from enum import Enum
from typing import Any, Dict

import penelope.utility as utility
from penelope.utility.file_utility import pickle_to_file, unpickle_from_file

from .container import InferredModel, TrainingCorpus

logger = utility.getLogger("")


class StoreCorpusOptions(Enum):
    NONE = 1
    PICKLED = 2


def _store_train_corpus(folder: str, train_corpus: TrainingCorpus, store_compressed: bool = True):

    """Stores the corpus used in training. If not pickled, then stored as separate files
    terms: Iterable[Iterable[str]]                               Never stored
    document_index: pd.DataFrame                                 TODO Stored as csv.zip
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
        document_index=train_corpus.document_index,
        id2word=train_corpus.id2word,
        vectorizer_args=train_corpus.vectorizer_args,
        corpus_options=train_corpus.corpus_options,
    )
    pickle_to_file(filename, _train_corpus)

    if _train_corpus.corpus_options is not None:
        store_corpus_options(folder=folder, options=_train_corpus.corpus_options)


def store_corpus_options(folder: str, options: Dict[str, Any]):
    filename = os.path.join(folder, "train_corpus_options.json")
    with open(filename, 'w') as fp:
        json.dump(options, fp, indent=4, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")


def load_corpus_options(folder: str) -> Dict[str, Any]:
    filename = os.path.join(folder, "train_corpus_options.json")
    if not os.path.isfile(filename):
        return None
    with open(filename, 'r') as f:
        options = json.load(f)
    return options


def _load_train_corpus(folder: str) -> TrainingCorpus:
    """Loads an train corpus from av previously pickled file."""
    filename = os.path.join(folder, "training_corpus.pickle.pbz2")
    if not os.path.isfile(filename):
        return None
    return unpickle_from_file(os.path.join(folder, "training_corpus.pickle.pbz2"))


def _store_topic_model(folder: str, topic_model: Any, store_compressed: bool = True):
    """Stores topic model in pickled format """
    filename = os.path.join(folder, f"topic_model.pickle{'.pbz2' if store_compressed else ''}")
    pickle_to_file(filename, topic_model)


def _load_topic_model(folder: str) -> Any:
    """Loads an train corpus from av previously pickled file."""
    return unpickle_from_file(os.path.join(folder, "topic_model.pickle.pbz2"))


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


def load_model(folder: str, lazy=True) -> InferredModel:
    """Loads inferred model data from previously pickled files."""
    topic_model = lambda: _load_topic_model(folder) if lazy else _load_topic_model(folder)
    train_corpus = lambda: _load_train_corpus(folder) if lazy else _load_train_corpus(folder)
    options = _load_model_options(folder)
    return InferredModel(topic_model=topic_model, train_corpus=train_corpus, **options)
