from __future__ import annotations

import abc
import os
import pickle
import sys
import types
from os.path import join as jj
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

import gensim
import numpy as np
import pandas as pd
import scipy
from penelope import utility
from penelope.corpus import DocumentIndex, DocumentIndexHelper, Token2Id, load_document_index
from tqdm.auto import tqdm

DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)

DocumentTopicsWeightsIter = Iterable[Tuple[int, Iterable[Tuple[int, float]]]]


class ITopicModelEngine(abc.ABC):
    def __init__(self, model: Any):
        self.model = model

    @staticmethod
    @abc.abstractmethod
    def is_supported(model: Any) -> bool:
        ...

    @staticmethod
    @abc.abstractmethod
    def supported_models() -> Sequence[Type]:
        ...

    @abc.abstractmethod
    def n_topics(self) -> int:
        ...

    @abc.abstractmethod
    def topics_tokens(self, n_tokens: int = 200, id2term: dict = None, **_) -> List[Tuple[float, str]]:
        ...

    @abc.abstractmethod
    def topic_tokens(self, topic_id: int, n_tokens: int = 200, id2term: dict = None, **_) -> List[Tuple[str, float]]:
        ...

    @staticmethod
    @abc.abstractmethod
    def train(
        train_corpus: "TrainingCorpus", method: str, engine_args: Dict[str, Any], **kwargs: Dict[str, Any]
    ) -> "InferredModel":
        ...

    @abc.abstractmethod
    def predict(self, corpus: Any, minimum_probability: float = 0.0, **kwargs) -> Iterable:
        ...

    def get_topic_token_weights(
        self, vocabulary: Any, n_tokens: int = 200, minimum_probability: float = 0.000001
    ) -> pd.DataFrame:
        """Compile document topic weights. Return DataFrame."""

        id2token: dict = Token2Id.any_to_id2token(vocabulary)
        topic_data = self.topics_tokens(n_tokens=n_tokens, id2term=id2token)

        topic_token_weights: pd.DataFrame = pd.DataFrame(
            [
                (topic_id, token, weight)
                for topic_id, tokens in topic_data
                for token, weight in tokens
                if weight > minimum_probability
            ],
            columns=['topic_id', 'token', 'weight'],
        )

        topic_token_weights['topic_id'] = topic_token_weights.topic_id.astype(np.uint16)

        fg = {v: k for k, v in id2token.items()}.get

        topic_token_weights['token_id'] = topic_token_weights.token.apply(fg)

        return topic_token_weights[['topic_id', 'token_id', 'token', 'weight']]

    def get_topic_token_overview(self, topic_token_weights: pd.DataFrame, n_tokens: int = 200) -> pd.DataFrame:
        """
        Group by topic_id and concatenate n_tokens words within group sorted by weight descending.
        There must be a better way of doing this...
        """

        alpha: List[float] = self.model.alpha if 'alpha' in self.model.__dict__ else None

        df = (
            topic_token_weights.groupby('topic_id')
            .apply(lambda x: sorted(list(zip(x["token"], x["weight"])), key=lambda z: z[1], reverse=True))
            .apply(lambda x: ' '.join([z[0] for z in x][:n_tokens]))
            .reset_index()
        )
        df.columns = ['topic_id', 'tokens']
        df['alpha'] = df.topic_id.apply(lambda topic_id: alpha[topic_id]) if alpha is not None else 0.0

        return df.set_index('topic_id')


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
        self._id2token: dict = None
        self._token2id: Token2Id = None

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

        os.makedirs(target_folder, exist_ok=True)

        if pickled:

            filename: str = jj(target_folder, "inferred_topics.pickle")

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
                archive_name = jj(target_folder, utility.replace_extension(name, ".zip"))
                utility.pandas_to_csv_zip(archive_name, (df, name), extension="csv", sep='\t')

    @staticmethod
    def load(*, folder: str, filename_fields: utility.FilenameFieldSpecs = None, pickled: bool = False):
        """Loads previously stored aggregate"""
        data = None

        if pickled:

            filename: str = jj(folder, "inferred_topics.pickle")

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
                    jj(folder, 'documents.zip'),
                    filename_fields=filename_fields,
                    sep='\t',
                    header=0,
                    index_col=0,
                    na_filter=False,
                ),
                dictionary=pd.read_csv(jj(folder, 'dictionary.zip'), sep='\t', header=0, index_col=0, na_filter=False),
                topic_token_weights=pd.read_csv(
                    jj(folder, 'topic_token_weights.zip'), sep='\t', header=0, index_col=0, na_filter=False
                ),
                topic_token_overview=pd.read_csv(
                    jj(folder, 'topic_token_overview.zip'), sep='\t', header=0, index_col=0, na_filter=False
                ),
                document_topic_weights=pd.read_csv(
                    jj(folder, 'document_topic_weights.zip'), sep='\t', header=0, index_col=0, na_filter=False
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
    def id2term(self) -> dict:
        if self._id2token is None:
            # id2token = inferred_topics.dictionary.to_dict()['token']
            self._id2token = self.dictionary.token.to_dict()
        return self._id2token

    @property
    def term2id(self) -> dict:
        return {v: k for k, v in self.id2term.items()}

    @property
    def token2id(self) -> Token2Id:
        if not self._token2id:
            self._token2id = Token2Id(data=self.term2id)
        return self._token2id

    @staticmethod
    def load_token2id(folder: str) -> Token2Id:
        dictionary: pd.DataFrame = pd.read_csv(
            jj(folder, 'dictionary.zip'), sep='\t', header=0, index_col=0, na_filter=False
        )
        data: dict = (
            dictionary.assign(token_id=dictionary.index)  # pylint: disable=no-member
            .set_index('token')
            .token_id.to_dict()
        )
        token2id: Token2Id = Token2Id(data=data)
        return token2id

    @staticmethod
    def compute_topic_proportions2(document_topic_weights: pd.DataFrame, doc_length_series: np.ndarray):
        """Compute topic proportations the LDAvis way. Fast version"""
        theta = scipy.sparse.coo_matrix(
            (document_topic_weights.weight, (document_topic_weights.document_id, document_topic_weights.topic_id))
        )
        theta_mult_doc_length = theta.T.multiply(doc_length_series).T
        topic_frequency = theta_mult_doc_length.sum(axis=0).A1
        topic_proportion = topic_frequency / topic_frequency.sum()
        return topic_proportion

    def compute_topic_proportions(self) -> pd.DataFrame:

        if 'n_terms' not in self.document_index.columns:
            return None

        _topic_proportions = InferredTopicsData.compute_topic_proportions2(
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
