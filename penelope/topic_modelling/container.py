import os
import pickle
import sys
import types
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import gensim
import pandas as pd
import penelope.utility as utility
import scipy

from .utility import add_document_metadata

logger = utility.getLogger('corpus_text_analysis')

DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)


class TrainingCorpus:
    """A container for the corpus data used during learning/inference"""

    def __init__(
        self,
        terms: Iterable[Iterable[str]] = None,
        documents: pd.DataFrame = None,
        doc_term_matrix: scipy.sparse.csr_matrix = None,
        id2word: Mapping[int, str] = None,
        vectorizer_args: Mapping[str, Any] = None,
        corpus: gensim.matutils.Sparse2Corpus = None,
    ):
        """A container for the corpus data used during learning/inference

        The corpus can be wither represented as a sequence of list of tokens or a docuement-term-matrix.

        The corpus actually used in training is stored in `corpus` by the modelling engine

        Parameters
        ----------
        terms : Iterable[Iterable[str]], optional
            Document tokens stream, by default None
        documents : pd.DataFrame, optional
            Documents metadata, by default None
        doc_term_matrix : scipy.sparse.csr_sparse, optional
            DTM BoW, by default None
        id2word : Union[gensim.corpora.Dictionary, Dict[int, str]], optional
            ID to word mapping, by default None
        vectorizer_args: Dict[str, Any]
            Options to use when vectorizing `terms`, ony used if DTM is None,
        """
        self.terms = terms
        self.doc_term_matrix = doc_term_matrix
        self.id2word = id2word
        self.documents = documents
        self.vectorizer_args = {**DEFAULT_VECTORIZE_PARAMS, **(vectorizer_args or {})}
        self.corpus = corpus


class InferredModel:
    """A container for the inferred model (the distributions over topic and word mixtures) during based on training data """

    def __init__(self, topic_model: Any, train_corpus: TrainingCorpus, method: str, **options: Dict[str, Any]):
        self.topic_model = topic_model
        self.train_corpus = train_corpus
        self.method = method
        self.options = options


class InferredTopicsData:
    """Container for a topic model as a generic set of pd.DataFrames. The content is common to all types model engines."""

    def __init__(
        self,
        documents: pd.DataFrame,  # documents (training, shuould be predicted?)
        dictionary: Any,  # dictionary
        topic_token_weights: pd.DataFrame,  # model data
        topic_token_overview: pd.DataFrame,  # model data
        document_topic_weights: pd.DataFrame,  # predicted data
    ):
        """A container for compiled data as generic pandas dataframes suitable for analysi and visualisation
        Parameters
        ----------
        documents : pd.DataFrame
            Corpus document index
        dictionary : Any
            Corpus dictionary
        topic_token_weights : pd.DataFrame
            Topic token weights
        topic_token_overview : pd.DataFrame
            Topic overview
        document_topic_weights : pd.DataFrame
            Document topic weights
        """
        self.dictionary = dictionary
        self.documents = documents
        self.topic_token_weights = topic_token_weights
        self.topic_token_overview = topic_token_overview
        self.document_topic_weights = document_topic_weights

        # Ensure that `year` column exists

        self.document_topic_weights = add_document_metadata(self.document_topic_weights, 'year', documents)

    @property
    def year_period(self) -> Tuple[int, int]:
        """Returns documents `year` interval (if exists)"""
        if 'year' not in self.document_topic_weights.columns:
            return (None, None)
        return (self.document_topic_weights.year.min(), self.document_topic_weights.year.max())

    @property
    def topic_ids(self) -> List[int]:
        """Returns unique topic ids """
        return list(self.document_topic_weights.topic_id.unique())

    def store(self, target_folder, pickled=False):
        """Stores aggregate in `target_folder` as individual zipped files

        Parameters
        ----------
        data_folder : str
            target folder
        model_name : str
            Model name
        pickled : bool, optional
            if True then pickled (binary) format else  CSV, by default False
        """

        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)

        if pickled:

            filename = os.path.join(target_folder, "inferred_topics.pickle")

            c_data = types.SimpleNamespace(
                documents=self.documents,
                dictionary=self.dictionary,
                topic_token_weights=self.topic_token_weights,
                topic_token_overview=self.topic_token_overview,
                document_topic_weights=self.document_topic_weights,
            )
            with open(filename, 'wb') as f:
                pickle.dump(c_data, f, pickle.HIGHEST_PROTOCOL)

        else:

            self.documents.to_csv(os.path.join(target_folder, 'documents.zip'), '\t')
            self.dictionary.to_csv(os.path.join(target_folder, 'dictionary.zip'), '\t')
            self.topic_token_weights.to_csv(os.path.join(target_folder, 'topic_token_weights.zip'), '\t')
            self.topic_token_overview.to_csv(os.path.join(target_folder, 'topic_token_overview.zip'), '\t')
            self.document_topic_weights.to_csv(os.path.join(target_folder, 'document_topic_weights.zip'), '\t')

    @staticmethod
    def load(folder: str, pickled: bool = False):
        """Loads previously stored aggregate"""
        data = None

        if pickled:

            filename = os.path.join(folder, "inferred_topics.pickle")

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            data = InferredTopicsData(
                documents=data.documents,
                dictionary=data.dictionary,
                topic_token_weights=data.topic_token_weights,
                topic_token_overview=data.topic_token_overview,
                document_topic_weights=data.document_topic_weights,
            )

        else:
            data = InferredTopicsData(
                documents=pd.read_csv(
                    os.path.join(folder, 'documents.zip'), '\t', header=0, index_col=0, na_filter=False
                ),
                dictionary=pd.read_csv(
                    os.path.join(folder, 'dictionary.zip'), '\t', header=0, index_col=0, na_filter=False
                ),
                topic_token_weights=pd.read_csv(
                    os.path.join(folder, 'topic_token_weights.zip'), '\t', header=0, index_col=0, na_filter=False
                ),
                topic_token_overview=pd.read_csv(
                    os.path.join(folder, 'topic_token_overview.zip'), '\t', header=0, index_col=0, na_filter=False
                ),
                document_topic_weights=pd.read_csv(
                    os.path.join(folder, 'document_topic_weights.zip'), '\t', header=0, index_col=0, na_filter=False
                ),
            )

        return data

    def info(self):
        for o_name in [k for k in self.__dict__ if not k.startswith("__")]:
            o_data = getattr(self, o_name)
            o_size = sys.getsizeof(o_data)
            print('{:>20s}: {:.4f} Mb {}'.format(o_name, o_size / (1024 * 1024), type(o_data)))

    @property
    def id2term(self):
        return self.dictionary.token.to_dict()

    @property
    def term2id(self):
        return {v: k for k, v in self.id2term.items()}
