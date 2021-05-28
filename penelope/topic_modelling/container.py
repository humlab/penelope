import os
import pickle
import sys
import types
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import gensim
import pandas as pd
import scipy
from penelope import utility
from penelope.corpus import DocumentIndex, DocumentIndexHelper, load_document_index
from penelope.utility import FilenameFieldSpecs, filename_utils
from tqdm.auto import tqdm

from .utility import compute_topic_proportions

DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)
jj = os.path.join


class TrainingCorpus:
    """A container for the corpus data used during learning/inference"""

    def __init__(
        self,
        terms: Iterable[Iterable[str]] = None,
        document_index: DocumentIndex = None,
        doc_term_matrix: scipy.sparse.csr_matrix = None,
        id2word: Optional[Mapping[int, str]] = None,
        vectorizer_args: Mapping[str, Any] = None,
        corpus: gensim.matutils.Sparse2Corpus = None,
        corpus_options: dict = None,
    ):
        """A container for the corpus data used during learning/inference

        The corpus can be wither represented as a sequence of list of tokens or a docuement-term-matrix.

        The corpus actually used in training is stored in `corpus` by the modelling engine

        Parameters
        ----------
        terms : Iterable[Iterable[str]], optional
            Document tokens stream, by default None
        document_index : DocumentIndex, optional
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
        self.corpus_options = corpus_options

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
            tbar = tqdm(desc="Lazy loading topic model...", position=0, leave=True)
            self._topic_model = self._topic_model()
            tbar.close()
        return self._topic_model

    @property
    def train_corpus(self):
        if callable(self._train_corpus):
            tbar = tqdm(desc="Lazy loading corpus...", position=0, leave=True)
            self._train_corpus = self._train_corpus()
            tbar.close()
        return self._train_corpus


class InferredTopicsData:
    """The result of applying a topic model on a corpus.
    The content, a generic set of pd.DataFrames, is common to all types of model engines.
    """

    def __init__(
        self,
        document_index: DocumentIndex,  # document_index (training, shuould be predicted?)
        dictionary: Any,  # dictionary
        topic_token_weights: pd.DataFrame,  # model data
        topic_token_overview: pd.DataFrame,  # model data
        document_topic_weights: pd.DataFrame,  # predicted data
    ):
        """A container for compiled data as generic pandas dataframes suitable for analysi and visualisation
        Parameters
        ----------
        document_index : DocumentIndex
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
        self.dictionary: Any = dictionary
        self.document_index: DocumentIndex = document_index
        self.topic_token_weights: pd.DataFrame = topic_token_weights
        self.document_topic_weights: pd.DataFrame = DocumentIndexHelper(document_index).overload(
            document_topic_weights, 'year'
        )
        self.topic_token_overview: pd.DataFrame = topic_token_overview
        self._id2token = None

    @property
    def num_topics(self) -> int:
        return int(self.topic_token_overview.index.max()) + 1

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

    def store(self, target_folder: str, pickled: bool = False):
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

            filename: str = os.path.join(target_folder, "inferred_topics.pickle")

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
                utility.pandas_to_csv_zip(archive_name, (df, name), extension="csv", sep='\t')

    @staticmethod
    def load(*, folder: str, filename_fields: FilenameFieldSpecs, pickled: bool = False):
        """Loads previously stored aggregate"""
        data = None

        if pickled:

            filename: str = os.path.join(folder, "inferred_topics.pickle")

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            data: InferredTopicsData = InferredTopicsData(
                document_index=data.document_index if hasattr(data, 'document_index') else data.document,
                dictionary=data.dictionary,
                topic_token_weights=data.topic_token_weights,
                topic_token_overview=data.topic_token_overview,
                document_topic_weights=data.document_topic_weights,
            )

        else:
            data: InferredTopicsData = InferredTopicsData(
                document_index=load_document_index(
                    os.path.join(folder, 'documents.zip'),
                    filename_fields=filename_fields,
                    sep='\t',
                    header=0,
                    index_col=0,
                    na_filter=False,
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

        assert "year" in data.document_index.columns

        return data

    def info(self):
        for o_name in [k for k in self.__dict__ if not k.startswith("__")]:
            o_data = getattr(self, o_name)
            o_size = sys.getsizeof(o_data)
            print('{:>20s}: {:.4f} Mb {}'.format(o_name, o_size / (1024 * 1024), type(o_data)))

    @property
    def id2term(self):
        if self._id2token is None:
            # id2token = inferred_topics.dictionary.to_dict()['token']
            self._id2token = self.dictionary.token.to_dict()
        return self._id2token

    @property
    def term2id(self):
        return {v: k for k, v in self.id2term.items()}

    # @property
    # def topic_proportions(self) -> pd.DataFrame:
    #     return compute_topic_proportions2(self.document_topic_weights)

    def compute_topic_proportions(self) -> pd.DataFrame:

        if 'n_terms' not in self.document_index.columns:
            return None

        _topic_proportions = compute_topic_proportions(
            self.document_topic_weights,
            self.document_index.n_terms.values,
        )

        return _topic_proportions

    def top_topic_token_weights_old(self, n_count: int) -> pd.DataFrame:
        id2token = self.id2term
        _largest = self.topic_token_weights.groupby(['topic_id'])[['topic_id', 'token_id', 'weight']].apply(
            lambda x: x.nlargest(n_count, columns=['weight'])
        )
        _largest['token'] = _largest.token_id.apply(lambda x: id2token[x])
        return _largest.set_index('topic_id')

    def top_topic_token_weights(self, n_count: int) -> pd.DataFrame:
        id2token = self.id2term
        _largest = (
            self.topic_token_weights.groupby(['topic_id'])[['topic_id', 'token_id', 'weight']]
            .apply(lambda x: x.nlargest(n_count, columns=['weight']))
            .reset_index(drop=True)
        )
        _largest['token'] = _largest.token_id.apply(lambda x: id2token[x])
        _largest['position'] = _largest.groupby('topic_id').cumcount() + 1
        return _largest.set_index('topic_id')
