from __future__ import annotations

import abc
import json
import os
import pickle
import sys
import types
from dataclasses import dataclass
from os.path import join as jj
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
from gensim.matutils import Sparse2Corpus
from gensim.models.tfidfmodel import TfidfModel
from penelope import utility
from penelope.corpus import DocumentIndex, DocumentIndexHelper, Token2Id, VectorizedCorpus, load_document_index
from penelope.vendor.gensim import terms_to_sparse_corpus
from scipy.sparse import coo_matrix, csr_matrix, issparse
from tqdm.auto import tqdm

CORPUS_OPTIONS_FILENAME: str = "train_corpus_options.json"
VECTORIZER_ARGS_FILENAME: str = "train_vectorizer_args.json"

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


@dataclass
class TrainingCorpus:
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
    id2token : Dict[int, str], optional
        ID to word mapping, by default None
    vectorizer_args: Dict[str, Any]
        Options to use when vectorizing `terms`, ony used if DTM is None,
    """

    terms: Iterable[Iterable[str]] = None
    doc_term_matrix: csr_matrix = None
    document_index: DocumentIndex = None
    id2token: Optional[Mapping[int, str]] = None

    vectorizer_args: Mapping[str, Any] = None
    corpus: Sparse2Corpus = None
    corpus_options: dict = None

    def __post_init__(self):
        self.vectorizer_args = {**DEFAULT_VECTORIZE_PARAMS, **(self.vectorizer_args or {})}

    @property
    def source(self) -> Any:
        return self.doc_term_matrix or self.terms

    def store(self, folder: str, store_compressed: bool = True, pickled: bool = True):
        """Stores the corpus used in training. If not pickled, then stored as separate files"""

        if pickled:

            filename = jj(folder, f"training_corpus.pickle{'.pbz2' if store_compressed else ''}")

            utility.pickle_to_file(
                filename,
                TrainingCorpus(
                    doc_term_matrix=None,
                    terms=None,
                    corpus=self.corpus,
                    document_index=self.document_index,
                    id2token=self.id2token,
                    vectorizer_args=self.vectorizer_args,
                    corpus_options=self.corpus_options,
                ),
            )

            utility.write_json(jj(folder, CORPUS_OPTIONS_FILENAME), data=self.corpus_options or {})

        else:

            utility.write_json(jj(folder, VECTORIZER_ARGS_FILENAME), data=self.vectorizer_args or {})
            utility.write_json(jj(folder, CORPUS_OPTIONS_FILENAME), data=self.corpus_options or {})

            corpus: VectorizedCorpus = VectorizedCorpus(
                self.corpus.sparse,
                token2id=self.id2token,
                document_index=self.document_index,
            )

            corpus.dump(tag='train', folder=folder)

    @staticmethod
    def load(folder: str) -> TrainingCorpus:
        """Loads an training corpus from pickled file."""
        for filename in ["training_corpus.pickle.pbz2", "training_corpus.pickle"]:
            if os.path.isfile(jj(folder, filename)):
                return utility.unpickle_from_file(jj(folder, filename))

        """Load from vectorized corpus if exists"""
        if VectorizedCorpus.dump_exists(tag='train', folder=folder):

            corpus: VectorizedCorpus = VectorizedCorpus.load(tag='train', folder=folder)
            return TrainingCorpus(
                corpus=None,  # Sparse2Corpus(corpus.data),
                doc_term_matrix=corpus.data,
                document_index=corpus.document_index,
                id2token=corpus.id2token,
                corpus_options=utility.read_json(jj(folder, CORPUS_OPTIONS_FILENAME), default={}),
                vectorizer_args=utility.read_json(jj(folder, VECTORIZER_ARGS_FILENAME), default={}),
            )
        return None

    def to_sparse_corpus(self) -> TrainingCorpus:
        """Create a Gensim Sparse2Corpus from `source`. Store in `corpus`."""
        if isinstance(self.source, Sparse2Corpus):

            return self

        if isinstance(self.source, VectorizedCorpus):

            self.corpus = Sparse2Corpus(self.source, documents_columns=False)
            self.id2token = self.source.id2token

        elif issparse(self.source):

            if self.id2token is None:
                raise ValueError("expected valid token2id, found None")

            self.corpus = Sparse2Corpus(self.source, documents_columns=False)

        else:

            self.corpus, self.id2token = terms_to_sparse_corpus(self.source)

        return self

    def to_tf_idf(self) -> TrainingCorpus:
        # assert algorithm_name != 'MALLETLDA', 'MALLET training model cannot (currently) use TFIDF weighed corpus'
        if self.corpus is None:
            raise ValueError("no corpus")
        tfidf_model = TfidfModel(self.corpus)
        self.corpus = [tfidf_model[d] for d in self.corpus]
        return self


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

    def store_topic_model(self, folder: str, store_compressed: bool = True):
        """Stores topic model in pickled format """
        filename: str = jj(folder, f"topic_model.pickle{'.pbz2' if store_compressed else ''}")
        os.makedirs(folder, exist_ok=True)
        utility.pickle_to_file(filename, self.topic_model)

    def store_model_options(self, folder: str):
        filename: str = jj(folder, "model_options.json")
        options: dict = {**dict(method=self.method), **self.options}
        os.makedirs(folder, exist_ok=True)
        with open(filename, 'w') as fp:
            json.dump(options, fp, indent=4, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")

    def store(self, folder: str, store_corpus=True, store_compressed=True):
        """Store model on disk in `folder`."""
        self.store_topic_model(folder, store_compressed=store_compressed)
        if store_corpus:
            self.train_corpus.store(folder, store_compressed=store_compressed)
        self.store_model_options(folder)

    @staticmethod
    def load_topic_model(folder: str) -> Any:
        """Load a topic model from pickled file."""
        for filename in ["topic_model.pickle.pbz2", "topic_model.pickle"]:
            if os.path.isfile(jj(folder, filename)):
                return utility.unpickle_from_file(jj(folder, filename))
        return None

    @staticmethod
    def load_model_options(folder: str) -> Dict[str, Any]:
        filename = jj(folder, "model_options.json")
        with open(filename, 'r') as f:
            options = json.load(f)
        return options

    @staticmethod
    def load(folder: str, lazy=True) -> InferredModel:
        """Load inferred model data from pickled files."""
        topic_model = lambda: InferredModel.load_topic_model(folder) if lazy else InferredModel.load_topic_model(folder)
        train_corpus = lambda: TrainingCorpus.load(folder) if lazy else TrainingCorpus.load(folder)
        options = InferredModel.load_model_options(folder)
        return InferredModel(topic_model=topic_model, train_corpus=train_corpus, **options)


class InferredTopicsData:
    """The result of applying a topic model on a corpus.
    The content, a generic set of pd.DataFrames, is common to all types of model engines.
    """

    def __init__(
        self,
        *,
        dictionary: pd.DataFrame,
        topic_token_weights: pd.DataFrame,
        topic_token_overview: pd.DataFrame,
        document_index: pd.DataFrame,
        document_topic_weights: pd.DataFrame,
    ):
        """A container for compiled data as generic pandas dataframes suitable for analysis and visualisation

        Model Args:
            dictionary (pd.DataFrame): Vocabulary
            topic_token_weights (pd.DataFrame): Topic token distributions
            topic_token_overview (pd.DataFrame): Topic token overview

        Predicted Args:
            document_index (pd.DataFrame): Documents index for predicted corpus.
            document_topic_weights (pd.DataFrame): Predicted weights
        """

        self.dictionary: pd.DataFrame = dictionary
        self.document_index: pd.DataFrame = document_index
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
        """Stores topics data in `target_folder` either as pickled file or individual zipped files """

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
        theta = coo_matrix(
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
