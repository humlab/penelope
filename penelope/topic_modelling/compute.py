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
    """Stores the inferred model on disk in folder `folder`

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

    if not store_corpus:
        inferred_model.train_corpus = None

    _store_model_pickled(folder, inferred_model)
    _store_model_options(folder, inferred_model)


def _store_train_corpus(folder: str, train_corpus: TrainingCorpus, pickled: bool=False):

    """Stores the corpus used in training. If not pickled, then stored as separate files
        terms: Iterable[Iterable[str]]                               Never stored
        documents: pd.DataFrame                                      Stored as csv.zip
        doc_term_matrix: scipy.sparse.csr_matrix                     Never stored
        id2word: Union[gensim.corpora.Dictionary, Dict[int, str]]    Stored compressed as gensim.Diciionary
        vectorizer_args: Dict[str, Any]                              Stored as json
        corpus: ???                                                  Stored as SparseCorpus
    """
    raise NotImplementedError()


def _store_model_pickled(folder, inferred_model):
    """Stores inferred model in pickled format

        topic_model: Gensim | MALLET | STTM     Always stored (as gensim or pickled if not gensim?)
        train_corpus: TrainingCorpus            Stored separately (optionally)
        method: str                             Stored in JSON infer.options.json: method
        options: Dict[str, Any]                 Stored in JSON infer.options.json: engine_args
    """
    filename = os.path.join(folder, "inferred_model.pickle")

    with open(filename, 'wb') as fp:
        pickle.dump(inferred_model, fp, pickle.HIGHEST_PROTOCOL)


def _store_model_options(folder, inferred_model):
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
