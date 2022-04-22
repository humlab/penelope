from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from os.path import isfile
from os.path import join as jj
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm

from penelope import corpus as pc
from penelope import utility
from penelope.vendor.gensim_api import corpora

from .topics_data import InferredTopicsData

CORPUS_OPTIONS_FILENAME: str = "train_corpus_options.json"
VECTORIZER_ARGS_FILENAME: str = "train_vectorizer_args.json"

DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)


class DeprecatedError(Exception):
    ...


AnyCorpus = Union[
    Iterable[Iterable[str]],
    Iterable[Tuple[str, Iterable[str]]],
    sp.spmatrix,
    pc.VectorizedCorpus,
    corpora.Sparse2Corpus,
    pc.TokenizedCorpus,
]


@dataclass
class TrainingCorpus:
    """A container for the corpus data used during training/inference

    Properties:
        corpus (AnyCorpus): Source corpus, can be almost anything. Defaults to None.
        document_index (DocumentIndex): Document's metadata. Defaults to None.
        token2id (Mapping[str, int]): Word to ID. Defaults to None.
        corpus_options (Dict[str, Any]): Options to use when vectorizing `corpus`. Defaults to None.
    """

    corpus: AnyCorpus = None
    document_index: pd.DataFrame = None
    token2id: pc.Token2Id = None

    vectorizer_args: Mapping[str, Any] = None
    corpus_options: dict = None

    def __post_init__(self):

        if isinstance(self.corpus, (pc.VectorizedCorpus, pc.TokenizedCorpus)):

            if not isinstance(self.token2id, pc.Token2Id):
                self.token2id = pc.Token2Id(data=self.corpus.token2id)

            self.document_index = self.corpus.document_index

        if isinstance(self.token2id, dict):
            self.token2id = pc.Token2Id(data=self.token2id)

        self.update_token_counts()

        self.vectorizer_args = {**DEFAULT_VECTORIZE_PARAMS, **(self.vectorizer_args or {})}

    @property
    def id2token(self) -> dict[int, str]:
        if self.token2id is None:
            return None
        if hasattr(self.token2id, 'id2token'):
            return self.token2id.id2token
        return {v: k for k, v in self.token2id.items()}

    def store(self, folder: str):
        """Stores the corpus used in training."""
        os.makedirs(folder, exist_ok=True)
        if isinstance(self.corpus, pc.VectorizedCorpus):
            corpus: pc.VectorizedCorpus = self.corpus
        elif isinstance(self.corpus, corpora.Sparse2Corpus):
            corpus: pc.VectorizedCorpus = pc.VectorizedCorpus(
                bag_term_matrix=self.corpus.sparse.tocsr().T,
                token2id=self.token2id,
                document_index=self.document_index,
            )
        else:
            raise NotImplementedError(f"type: {type(self.corpus)} save not implemented")

        assert len(self.document_index) == corpus.data.shape[0], 'bug check: corpus transpose needed?'

        os.makedirs(folder, exist_ok=True)
        utility.write_json(jj(folder, VECTORIZER_ARGS_FILENAME), data=self.vectorizer_args or {})
        utility.write_json(jj(folder, CORPUS_OPTIONS_FILENAME), data=self.corpus_options or {})

        corpus.dump(tag='train', folder=folder)

    @property
    def document_token_counts(self):
        if isinstance(self.corpus, pc.VectorizedCorpus):
            return self.corpus.document_token_counts
        if isinstance(self.corpus, corpora.Sparse2Corpus):
            return self.corpus.sparse.sum(axis=0).A1
        return ValueError(f"expected sparse corpus, found {type(self.corpus)}")

    def update_token_counts(self) -> TrainingCorpus:
        """logs doc token count in document index (if missing)"""
        if self.document_index is not None:
            self.document_index: pd.DataFrame = pc.update_document_index_token_counts_by_corpus(
                self.document_index, self.corpus
            )
        return self

    @staticmethod
    def exists(folder: str) -> bool:
        if folder is None:
            return False
        return pc.VectorizedCorpus.dump_exists(tag='train', folder=folder)

    @staticmethod
    def load(folder: str) -> TrainingCorpus:
        """Loads an training corpus from pickled file."""

        """Load from vectorized corpus if exists"""
        if pc.VectorizedCorpus.dump_exists(tag='train', folder=folder):
            corpus: pc.VectorizedCorpus = pc.VectorizedCorpus.load(tag='train', folder=folder)
            return TrainingCorpus(
                corpus=corpus,
                document_index=corpus.document_index,
                token2id=pc.Token2Id(data=corpus.token2id),
                corpus_options=utility.read_json(jj(folder, CORPUS_OPTIONS_FILENAME), default={}),
                vectorizer_args=utility.read_json(jj(folder, VECTORIZER_ARGS_FILENAME), default={}),
            )

        return None


class InferredModel:
    """A container for the trained topic model"""

    OPTIONS_FILENAME: str = "model_options.json"
    ID2TOKEN_FILENAME: str = "topic_model_id2token.json.gz"
    MODEL_FILENAMES: List[str] = ["topic_model.pickle.pbz2", "topic_model.pickle"]

    def __init__(self, topic_model: Any, id2token: Mapping[int, str], options: Dict[str, Any]):
        self._topic_model: Any = topic_model
        self.id2token: Mapping[int, str] = id2token
        self.method: str = options.get('method')
        self.options: dict = options

    @property
    def topic_model(self) -> Any:
        if callable(self._topic_model):
            tbar = tqdm(desc="Lazy loading topic model...", position=0, leave=True)
            self._topic_model = self._topic_model()
            tbar.close()
        return self._topic_model

    @staticmethod
    def exists(folder: str) -> bool:
        return isfile(jj(folder, InferredModel.OPTIONS_FILENAME)) and any(
            isfile(jj(folder, x)) for x in InferredModel.MODEL_FILENAMES
        )

    @staticmethod
    def load(folder: str, lazy=True) -> InferredModel:
        """Load inferred model data from pickled files."""
        topic_model = lambda: InferredModel.load_model(folder) if lazy else InferredModel.load_model(folder)
        options: dict = InferredModel.load_options(folder)
        id2token: Mapping[int, str] = InferredModel.load_id2token(folder)
        return InferredModel(topic_model=topic_model, id2token=id2token, options=options)

    def store(self, folder: str, store_compressed=True):
        """Store model on disk in `folder`."""
        self.store_model(folder, store_compressed=store_compressed)
        self.store_id2token(folder)
        self.store_options(folder)

    @staticmethod
    def load_model(folder: str) -> Any:
        """Load a topic model from pickled file."""
        if not InferredModel.exists(folder):
            raise FileNotFoundError(f"no model found in folder {folder}")
        for filename in ["topic_model.pickle.pbz2", "topic_model.pickle"]:
            if isfile(jj(folder, filename)):
                return utility.unpickle_from_file(jj(folder, filename))
        return None

    def store_model(self, folder: str, store_compressed: bool = True):
        """Stores topic model in pickled format"""
        filename: str = jj(folder, f"topic_model.pickle{'.pbz2' if store_compressed else ''}")
        os.makedirs(folder, exist_ok=True)
        utility.pickle_to_file(filename, self.topic_model)

    @staticmethod
    def load_id2token(folder: str) -> Mapping[int, str]:
        """Loads vocabulary from file"""
        filename: str = jj(folder, InferredModel.ID2TOKEN_FILENAME)
        if not os.path.isfile(filename):
            """Backward compatibility: read dictionary.zip"""
            return InferredTopicsData.load_token2id(folder).id2token
        with gzip.open(filename, 'rb') as f:
            json_str = f.read().decode(encoding='utf-8')
            return {int(k): v for k, v in json.loads(json_str).items()}

    def store_id2token(self, folder: str) -> Mapping[int, str]:
        """Stores vocabulary in json format"""
        filename: str = jj(folder, InferredModel.ID2TOKEN_FILENAME)
        json_bytes: bytes = json.dumps(self.id2token).encode('utf-8')
        with gzip.open(filename, 'wb') as f:
            f.write(json_bytes)

    @staticmethod
    def load_options(folder: str) -> Dict[str, Any]:
        filename = jj(folder, InferredModel.OPTIONS_FILENAME)
        with open(filename, 'r') as f:
            options = json.load(f)
        return options

    def store_options(self, folder: str):
        filename: str = jj(folder, InferredModel.OPTIONS_FILENAME)
        options: dict = {'method': self.method, **self.options}
        os.makedirs(folder, exist_ok=True)
        with open(filename, 'w') as fp:
            json.dump(options, fp, indent=4, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")
