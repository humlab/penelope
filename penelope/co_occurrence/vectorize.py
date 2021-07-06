import array
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import IntEnum
from pprint import pformat as pf
from typing import Any, Iterable, Iterator, Mapping, Optional, Set

import numpy as np
import scipy
from loguru import logger

DEBUG_TRACE: bool = False
if DEBUG_TRACE:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO", enqueue=True)
    logger.add("co_occurrence_trace.py", rotation=None, format="{message}", serialize=False, level="INFO", enqueue=True)


class VectorizeType(IntEnum):
    Normal = 1
    Concept = 2


@dataclass
class VectorizedTTM:
    vectorize_type: VectorizeType
    term_term_matrix: scipy.sparse.spmatrix
    term_window_counts: Mapping[int, int]
    document_id: int

    def trace(self):
        logger.info(
            "\n#################################################################################################"
        )
        logger.info(f"# VectorizedTTM state #{self.document_id}")
        logger.info(
            "#################################################################################################\n"
        )
        logger.info(f"document_id = {self.document_id}")
        logger.info(f"vectorize_type = VectorizeType.{self.vectorize_type.name}")
        logger.info(f"term_term_matrix = {pf(self.term_term_matrix.todense())}")
        logger.info(f"term_window_counts = {pf(self.term_window_counts)}")


def windows_to_ttm(
    *,
    document_id: int,
    windows: Iterator[Iterable[int]],
    concept_ids: Set[int],
    ignore_ids: Set[int],
    vocab_size: int,
) -> Mapping[VectorizeType, VectorizedTTM]:

    if len(concept_ids) > 1:
        raise NotImplementedError("Multiple concepts disabled (performance")

    concept_id: Optional[str] = list(concept_ids)[0] if concept_ids else None

    def _count_tokens_without_ignores(window: Iterable[int]) -> dict:
        token_counter: dict = {}
        tg = token_counter.get
        for t in window:
            token_counter[t] = tg(t, 0) + 1
        return token_counter

    def _count_tokens_with_ignores(window: Iterable[str]) -> dict:
        token_counter: dict = {}
        tg = token_counter.get
        for t in window:
            if t in ignore_ids:
                continue
            token_counter[t] = tg(t, 0) + 1
        return token_counter

    counters: Mapping[VectorizeType, WindowsTermsCounter] = defaultdict(WindowsTermsCounter)
    count_tokens = _count_tokens_with_ignores if ignore_ids else _count_tokens_without_ignores

    ewu = counters[VectorizeType.Normal].update
    if concept_id is not None:
        cwu = counters[VectorizeType.Concept].update
        for window in windows:
            token_counts: dict = count_tokens(window)
            if concept_id in window:  # any(x in window for x in concept):
                cwu(token_counts)
            ewu(token_counts)
    else:
        logger.info("no concept")
        for window in windows:
            ewu(count_tokens(window))

    data: Mapping[VectorizeType, VectorizedTTM] = {
        key: counter.compile(vectorize_type=key, document_id=document_id, vocab_size=vocab_size)
        for key, counter in counters.items()
    }

    return data


class WindowsTermsCounter:
    """Contains term window counts collected during TTM construction.
    Compiles the stored term window counts into a TTM matrix."""

    def __init__(self, dtype: Any = np.int32):
        self.dtype = dtype
        self.indptr = []
        self.jj = []
        self.values = array.array(str("i"))
        self.indptr.append(0)
        self._jj_extend = self.jj.extend
        self._values_extend = self.values.extend
        self._indptr_append = self.indptr.append

    def update(self, token_counter: Counter):

        if None in token_counter.keys():
            raise ValueError("BugCheck: None in TokenCounter not allowed!")

        self._jj_extend(token_counter.keys())
        self._values_extend(token_counter.values())
        self._indptr_append(len(self.jj))

    def compile(self, *, document_id: int, vectorize_type: VectorizeType, vocab_size: int) -> VectorizedTTM:
        """Computes the final TTM and a global term window counter"""
        self.jj = np.asarray(self.jj, dtype=np.int64)
        self.indptr = np.asarray(self.indptr, dtype=np.int32)
        self.values = np.frombuffer(self.values, dtype=np.intc)

        window_term_matrix: scipy.sparse.spmatrix = scipy.sparse.csr_matrix(
            (self.values, self.jj, self.indptr), shape=(len(self.indptr) - 1, vocab_size), dtype=self.dtype
        )
        window_term_matrix.sort_indices()

        term_term_matrix: scipy.sparse.spmatrix = scipy.sparse.triu(
            np.dot(window_term_matrix.T, window_term_matrix),
            1,
        )
        term_window_counts: Mapping[int, int] = self._to_term_window_counter(window_term_matrix)
        return VectorizedTTM(
            vectorize_type=vectorize_type,
            document_id=document_id,
            term_term_matrix=term_term_matrix,
            term_window_counts=term_window_counts,
        )

    def _to_term_window_counter(self, window_term_matrix: scipy.sparse.spmatrix) -> Mapping[int, int]:
        """Returns tuples (token_id, window count) for non-zero tokens in window_term_matrix"""

        window_counts: np.ndarray = (window_term_matrix != 0).sum(axis=0).A1
        window_counter: Mapping[int, int] = {i: window_counts[i] for i in window_counts.nonzero()[0]}
        return window_counter
