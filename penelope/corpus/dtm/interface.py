# type: ignore

import abc
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy

from ..document_index import DocumentIndex


class VectorizedCorpusError(ValueError): ...


# pylint: disable=too-many-public-methods
class IVectorizedCorpus(abc.ABC):
    @property
    @abc.abstractmethod
    def token2id(self) -> dict[str, int]: ...

    @property
    @abc.abstractmethod
    def bag_term_matrix(self) -> scipy.sparse.csr_matrix: ...

    @property
    @abc.abstractmethod
    def id2token(self) -> dict[int, str]: ...

    @property
    @abc.abstractmethod
    def vocabulary(self) -> List[str]: ...

    @abc.abstractmethod
    def nlargest(self, n_top: int, sort_indices: bool = False, override: bool = False) -> np.ndarray: ...

    @property
    @abc.abstractmethod
    def overridden_term_frequency(self) -> Dict[str, int]: ...

    @property
    @abc.abstractmethod
    def term_frequency(self) -> np.ndarray: ...

    @property
    @abc.abstractmethod
    def term_frequency0(self) -> np.ndarray: ...

    @property
    @abc.abstractmethod
    def document_token_counts(self) -> np.ndarray: ...

    @property
    @abc.abstractmethod
    def T(self) -> scipy.sparse.csr_matrix: ...

    @property
    @abc.abstractmethod
    def data(self) -> scipy.sparse.csr_matrix: ...

    @property
    @abc.abstractmethod
    def n_docs(self) -> int: ...

    @property
    @abc.abstractmethod
    def n_tokens(self) -> int: ...

    @property
    @abc.abstractmethod
    def document_index(self) -> DocumentIndex: ...

    @property
    @abc.abstractmethod
    def payload(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    def todense(self) -> "IVectorizedCorpus": ...

    @abc.abstractmethod
    def dump(self, *, tag: str, folder: str, compressed: bool = True) -> "IVectorizedCorpus": ...

    @staticmethod
    @abc.abstractmethod
    def dump_exists(*, tag: str, folder: str) -> bool: ...

    @staticmethod
    @abc.abstractmethod
    def remove(*, tag: str, folder: str): ...

    @staticmethod
    @abc.abstractmethod
    def load(*, tag: str, folder: str) -> "IVectorizedCorpus": ...

    @staticmethod
    @abc.abstractmethod
    def dump_options(*, tag: str, folder: str, options: dict): ...

    @staticmethod
    @abc.abstractmethod
    def load_options(*, tag: str, folder: str) -> dict: ...

    @abc.abstractmethod
    def filter(self, px) -> "IVectorizedCorpus": ...

    @abc.abstractmethod
    def normalize(self, axis: int = 1, norm: str = 'l1', keep_magnitude: bool = False) -> "IVectorizedCorpus": ...

    @abc.abstractmethod
    def normalize_by_raw_counts(self) -> "IVectorizedCorpus": ...

    @abc.abstractmethod
    def pick_top_tf_map(self, n_top: int) -> Dict[str, int]: ...

    @abc.abstractmethod
    def slice_by_tf(self, threshold: int | None) -> "IVectorizedCorpus": ...

    @abc.abstractmethod
    def slice_by_n_top(self, n_top: int | None) -> "IVectorizedCorpus": ...

    @abc.abstractmethod
    def slice_by_document_frequency(self, max_df=1.0, min_df=1, max_n_terms=None) -> "IVectorizedCorpus": ...

    @abc.abstractmethod
    def slice_by(self, px) -> "IVectorizedCorpus": ...

    @abc.abstractmethod
    def translate_to_vocab(self, id2token: dict[int, str], inplace=False) -> "IVectorizedCorpus": ...

    @abc.abstractmethod
    def stats(self): ...

    @abc.abstractmethod
    def to_n_top_dataframe(self, n_top: int): ...

    @abc.abstractmethod
    def token_indices(self, tokens: Iterable[str]) -> list[int]: ...

    @abc.abstractmethod
    def tf_idf(self, norm: str = 'l2', use_idf: bool = True, smooth_idf: bool = True) -> "IVectorizedCorpus": ...

    @abc.abstractmethod
    def to_bag_of_terms(self, indices: Optional[Iterable[int]] = None) -> Iterable[Iterable[str]]: ...

    @abc.abstractmethod
    def get_top_n_words(self, n: int = 1000, indices: Sequence[int] = None) -> Sequence[Tuple[str, Number]]: ...

    @abc.abstractmethod
    def get_partitioned_top_n_words(
        self,
        category_column: str = 'category',
        n_top: int = 100,
        pad: str = None,
        keep_empty: bool = False,
    ) -> dict: ...

    @abc.abstractmethod
    def get_top_terms(
        self,
        category_column: str = 'category',
        n_top: int = 100,
        kind: str = 'token',
    ) -> pd.DataFrame: ...

    @abc.abstractmethod
    def co_occurrence_matrix(self) -> scipy.sparse.spmatrix: ...

    @abc.abstractmethod
    def find_matching_words(self, word_or_regexp: List[str], n_max_count: int, descending: bool) -> List[str]: ...

    @abc.abstractmethod
    def find_matching_words_indices(
        self, word_or_regexp: List[str], n_max_count: int, descending: bool
    ) -> List[int]: ...

    @abc.abstractmethod
    def pick_n_top_words(self, words: List[str], n_top: int, descending: bool) -> List[str]: ...

    @abc.abstractmethod
    def zero_out_by_tf_threshold(self, tf_threshold: Union[int, float]) -> Sequence[int]: ...

    @abc.abstractmethod
    def zero_out_by_indices(self, indices: Sequence[int]) -> Sequence[int]: ...

    @staticmethod
    @abc.abstractmethod
    def create(
        bag_term_matrix: scipy.sparse.csr_matrix,
        token2id: Dict[str, int],
        document_index: DocumentIndex,
        overridden_term_frequency: Dict[str, int] = None,
    ) -> "IVectorizedCorpus": ...


class IVectorizedCorpusProtocol(Protocol):
    @staticmethod
    def create(
        bag_term_matrix: scipy.sparse.csr_matrix,
        token2id: Dict[str, int],
        document_index: DocumentIndex,
        overridden_term_frequency: Dict[str, int] = None,
        **kwargs,
    ) -> IVectorizedCorpus: ...

    @property
    def term_frequency(self) -> np.ndarray: ...

    @property
    def term_frequency0(self) -> np.ndarray: ...

    @property
    def document_index(self) -> DocumentIndex: ...

    @property
    def bag_term_matrix(self) -> scipy.sparse.csr_matrix: ...

    @property
    def id2token(self) -> dict[int, str]: ...

    @property
    def token2id(self) -> dict[str, int]: ...

    @property
    def payload(self) -> dict[str, Any]: ...

    def remember(self, **kwargs) -> None: ...

    def recall(self, key: str) -> Optional[Any]: ...

    def nlargest(self, n_top: int, *, sort_indices: bool = False, override: bool = False) -> np.ndarray: ...
