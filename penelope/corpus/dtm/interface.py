# type: ignore

import abc
from numbers import Number
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy

from ..document_index import DocumentIndex


class VectorizedCorpusError(ValueError):
    ...


# pylint: disable=too-many-public-methods
class IVectorizedCorpus(abc.ABC):
    @property
    @abc.abstractproperty
    def token2id(self) -> Mapping[int, str]:
        ...

    @property
    @abc.abstractproperty
    def bag_term_matrix(self) -> scipy.sparse.csr_matrix:
        ...

    @property
    @abc.abstractproperty
    def id2token(self) -> Mapping[str, int]:
        ...

    @property
    @abc.abstractproperty
    def vocabulary(self) -> List[str]:
        ...

    @property
    @abc.abstractproperty
    def term_frequency_mapping(self) -> Dict[str, int]:
        ...

    @property
    @abc.abstractproperty
    def term_frequencies(self) -> np.ndarray:
        ...

    @property
    @abc.abstractproperty
    def document_token_counts(self) -> np.ndarray:
        ...

    @property
    @abc.abstractproperty
    def T(self) -> scipy.sparse.csr_matrix:
        ...

    @property
    @abc.abstractproperty
    def data(self) -> scipy.sparse.csr_matrix:
        ...

    @property
    @abc.abstractproperty
    def n_docs(self) -> int:
        ...

    @property
    @abc.abstractproperty
    def n_terms(self) -> int:
        ...

    @property
    @abc.abstractproperty
    def document_index(self) -> DocumentIndex:
        ...

    @property
    @abc.abstractproperty
    def payload(self) -> Mapping[str, Any]:
        ...

    @abc.abstractmethod
    def todense(self) -> "IVectorizedCorpus":
        ...

    @abc.abstractmethod
    def dump(self, *, tag: str, folder: str, compressed: bool = True) -> "IVectorizedCorpus":
        ...

    @staticmethod
    @abc.abstractmethod
    def dump_exists(*, tag: str, folder: str) -> bool:
        ...

    @staticmethod
    @abc.abstractmethod
    def remove(*, tag: str, folder: str):
        ...

    @staticmethod
    @abc.abstractmethod
    def load(*, tag: str, folder: str) -> "IVectorizedCorpus":
        ...

    @staticmethod
    @abc.abstractmethod
    def dump_options(*, tag: str, folder: str, options: dict):
        ...

    @staticmethod
    @abc.abstractmethod
    def load_options(*, tag: str, folder: str) -> dict:
        ...

    @abc.abstractmethod
    def filter(self, px) -> "IVectorizedCorpus":
        ...

    @abc.abstractmethod
    def normalize(self, axis: int = 1, norm: str = 'l1', keep_magnitude: bool = False) -> "IVectorizedCorpus":
        ...

    @abc.abstractmethod
    def normalize_by_raw_counts(self) -> "IVectorizedCorpus":
        ...

    @abc.abstractmethod
    def n_global_top_tokens(self, n_top: int) -> Dict[str, int]:
        ...

    @abc.abstractmethod
    def slice_by_n_count(self, n_count: int) -> "IVectorizedCorpus":
        ...

    @abc.abstractmethod
    def slice_by_n_top(self, n_top) -> "IVectorizedCorpus":
        ...

    @abc.abstractmethod
    def slice_by_document_frequency(self, max_df=1.0, min_df=1, max_n_terms=None) -> "IVectorizedCorpus":
        ...

    @abc.abstractmethod
    def slice_by(self, px) -> "IVectorizedCorpus":
        ...

    @abc.abstractmethod
    def stats(self):
        ...

    @abc.abstractmethod
    def to_n_top_dataframe(self, n_top: int):
        ...

    @abc.abstractmethod
    def year_range(self) -> Tuple[Optional[int], Optional[int]]:
        ...

    @abc.abstractmethod
    def xs_years(self) -> Tuple[int, int]:
        ...

    @abc.abstractmethod
    def token_indices(self, tokens: Iterable[str]):
        ...

    @abc.abstractmethod
    def tf_idf(self, norm: str = 'l2', use_idf: bool = True, smooth_idf: bool = True) -> "IVectorizedCorpus":
        ...

    @abc.abstractmethod
    def to_bag_of_terms(self, indicies: Optional[Iterable[int]] = None) -> Iterable[Iterable[str]]:
        ...

    @abc.abstractmethod
    def get_top_n_words(self, n: int = 1000, indices: Sequence[int] = None) -> Sequence[Tuple[str, Number]]:
        ...

    @abc.abstractmethod
    def get_partitioned_top_n_words(
        self,
        category_column: str = 'category',
        n_count: int = 100,
        pad: str = None,
        keep_empty: bool = False,
    ) -> dict:
        ...

    @abc.abstractmethod
    def get_top_terms(
        self,
        category_column: str = 'category',
        n_count: int = 100,
        kind: str = 'token',
    ) -> pd.DataFrame:
        ...

    @abc.abstractmethod
    def co_occurrence_matrix(self) -> scipy.sparse.spmatrix:
        ...

    @abc.abstractmethod
    def find_matching_words(self, word_or_regexp: List[str], n_max_count: int, descending: bool) -> List[str]:
        ...

    @abc.abstractmethod
    def find_matching_words_indices(self, word_or_regexp: List[str], n_max_count: int, descending: bool) -> List[int]:
        ...

    @abc.abstractmethod
    def pick_n_top_words(self, words: List[str], n_top: int, descending: bool) -> List[str]:
        ...

    @staticmethod
    @abc.abstractmethod
    def create(
        bag_term_matrix: scipy.sparse.csr_matrix,
        token2id: Dict[str, int],
        document_index: DocumentIndex,
        term_frequency_mapping: Dict[str, int] = None,
    ) -> "IVectorizedCorpus":
        ...


class IVectorizedCorpusProtocol(Protocol):
    ...

    @staticmethod
    def create(
        bag_term_matrix: scipy.sparse.csr_matrix,
        token2id: Dict[str, int],
        document_index: DocumentIndex,
        term_frequency_mapping: Dict[str, int] = None,
    ) -> IVectorizedCorpus:
        ...

    @property
    def document_index(self) -> DocumentIndex:
        ...

    @property
    def bag_term_matrix(self) -> scipy.sparse.csr_matrix:
        ...

    @property
    def id2token(self) -> Mapping[int, str]:
        ...

    @property
    def token2id(self) -> Mapping[str, int]:
        ...

    @property
    def payload(self) -> Mapping[str, Any]:
        ...
