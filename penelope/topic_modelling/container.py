import os
import pickle
import sys
import types
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import gensim
import pandas as pd
import penelope.utility as utility
import scipy
from penelope.corpus.document_index import document_index_upgrade
from penelope.utility import file_utility, filename_utils
from tqdm.auto import tqdm

from .utility import add_document_metadata

logger = utility.getLogger('corpus_text_analysis')

DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)
jj = os.path.join


class TrainingCorpus:
    """A container for the corpus data used during learning/inference"""

    def __init__(
        self,
        terms: Iterable[Iterable[str]] = None,
        document_index: pd.DataFrame = None,
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
        document_index : pd.DataFrame, optional
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
        self.documents = document_index
        self.vectorizer_args = {**DEFAULT_VECTORIZE_PARAMS, **(vectorizer_args or {})}
        self.corpus = corpus

    @property
    def document_index(self):
        return self.documents


class InferredModel:
    """A container for the trained topic model """

    def __init__(
        self,
        topic_model: Any,
        train_corpus: TrainingCorpus,
        method: str,
        **options: Dict[str, Any],
    ):
        self._topic_model = topic_model
        self._train_corpus = train_corpus
        self.method = method
        self.options = options

    @property
    def topic_model(self):
        if callable(self._topic_model):
            tbar = tqdm(desc="Lazy Loading model...", position=0, leave=True)
            self._topic_model = self._topic_model()
            tbar.close()
        return self._topic_model

    @property
    def train_corpus(self):
        if callable(self._train_corpus):
            tbar = tqdm(desc="Lazy corpus...", position=0, leave=True)
            self._train_corpus = self._train_corpus()
            tbar.close()
        return self._train_corpus


class InferredTopicsData:
    """The result of applying a topic model on a corpus.
    The content, a generic set of pd.DataFrames, is common to all types model engines.
    """

    def __init__(
        self,
        document_index: pd.DataFrame,  # document_index (training, shuould be predicted?)
        dictionary: Any,  # dictionary
        topic_token_weights: pd.DataFrame,  # model data
        topic_token_overview: pd.DataFrame,  # model data
        document_topic_weights: pd.DataFrame,  # predicted data
    ):
        """A container for compiled data as generic pandas dataframes suitable for analysi and visualisation
        Parameters
        ----------
        document_index : pd.DataFrame
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
        self.document_index = document_index
        self.topic_token_weights = topic_token_weights
        self.topic_token_overview = topic_token_overview
        self.document_topic_weights = document_topic_weights

        # Ensure that `year` column exists

        self.document_topic_weights = add_document_metadata(self.document_topic_weights, 'year', document_index)

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
                documents=self.document_index,
                dictionary=self.dictionary,
                topic_token_weights=self.topic_token_weights,
                topic_token_overview=self.topic_token_overview,
                document_topic_weights=self.document_topic_weights,
            )
            with open(filename, 'wb') as f:
                pickle.dump(c_data, f, pickle.HIGHEST_PROTOCOL)

        else:
            data = [
                (self.document_index.rename_axis(''), 'documents.csv'),
                (self.dictionary, 'dictionary.csv'),
                (self.topic_token_weights, 'topic_token_weights.csv'),
                (self.topic_token_overview, 'topic_token_overview.csv'),
                (self.document_topic_weights, 'document_topic_weights.csv'),
            ]

            for (df, name) in data:
                archive_name = jj(target_folder, filename_utils.replace_extension(name, ".zip"))
                file_utility.pandas_to_csv_zip(archive_name, (df, name), extension="csv", sep='\t')

    @staticmethod
    def load(folder: str, pickled: bool = False):
        """Loads previously stored aggregate"""
        data = None

        if pickled:

            filename = os.path.join(folder, "inferred_topics.pickle")

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            data = InferredTopicsData(
                document_index=data.document_index if hasattr(data, 'document_index') else data.document,
                dictionary=data.dictionary,
                topic_token_weights=data.topic_token_weights,
                topic_token_overview=data.topic_token_overview,
                document_topic_weights=data.document_topic_weights,
            )

        else:
            data = InferredTopicsData(
                document_index=pd.read_csv(
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

            data.document_index = document_index_upgrade(data.document_index)

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
