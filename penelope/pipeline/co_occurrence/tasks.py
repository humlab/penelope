import collections
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import numpy as np
import scipy
import scipy.sparse as sp
from penelope.co_occurrence import (
    Bundle,
    ContextOpts,
    CoOccurrenceError,
    TokenWindowCountStatistics,
    WindowsCoOccurrenceOutput,
    WindowsCoOccurrenceVectorizer,
    tokens_to_windows,
)
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.corpus.dtm import to_word_pair_token
from penelope.pipeline.tasks_mixin import VocabularyIngestMixIn
from penelope.type_alias import DocumentIndex

from ..interfaces import ContentType, DocumentPayload, ITask, PipelineError


@dataclass
class CoOccurrencePayload:
    document_id: int
    term_term_matrix: scipy.sparse.spmatrix
    term_window_counter: Mapping[int, int]


def TTM_to_co_occurrence_DTM(
    stream: Iterable[CoOccurrencePayload], token2id: Token2Id, document_index: DocumentIndex
) -> VectorizedCorpus:
    """Tranforms a sequence of document-wise term-term matrices to a corpus-wide document-term matrix"""

    """NOTE: This implementation depends on stream being reiterable..."""

    """Ingest token-pairs into new COO-vocabulary using existing token vocabulary"""
    vocabulary: Token2Id = Token2Id()

    fg: Callable[[int], str] = token2id.id2token.get

    for item in stream:
        TTM: scipy.sparse.spmatrix = item.term_term_matrix
        vocabulary.ingest(to_word_pair_token(a, b, fg) for (a, b) in zip(TTM.row, TTM.col))

    vocabulary.close()

    """Create sparse matrix where rows are documents, and columns are "token-pairs" tokens"""
    matrix: scipy.sparse.lil_matrix = scipy.sparse.lil_matrix((len(document_index), len(vocabulary)), dtype=int)

    for item in stream:
        TTM: scipy.sparse.spmatrix = item.term_term_matrix

        """Translate token-pair ids into id in new COO-vocabulary"""
        token_ids = [vocabulary[to_word_pair_token(a, b, fg)] for (a, b) in zip(TTM.row, TTM.col)]

        matrix[item.document_id, [token_ids]] = TTM.data

    document_index = document_index.set_index('document_id', drop=False)

    corpus = VectorizedCorpus(
        bag_term_matrix=matrix.tocsr(), token2id=dict(vocabulary.data), document_index=document_index
    )

    return corpus


@dataclass
class ToCoOccurrenceDTM(VocabularyIngestMixIn, ITask):
    """Computes (DOCUMENT-LEVEL) windows co-occurrence.

    Bundle consists of the following document level information:

        1) Co-occurrence matrix (TTM) with number of common windows in document
        2) Mapping with number of windows each term occurs in

    Iterable[DocumentPayload] => Iterable[DocumentPayload]
        DocumentPayload.content = Tuple[document_id, TTM, token_window_counts]

    """

    context_opts: ContextOpts = None
    vectorizer: WindowsCoOccurrenceVectorizer = field(init=False, default=None)

    def __post_init__(self):
        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.CO_OCCURRENCE_DTM_DOCUMENT

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("context_opts", self.context_opts)
        return self

    def enter(self):
        super().enter()

        if self.pipeline.payload.token2id is None:
            raise PipelineError(f"{type(self).__name__} requires a vocabulary!")

        if self.context_opts.pad not in self.token2id:
            _ = self.token2id[self.context_opts.pad]

        self.vectorizer: WindowsCoOccurrenceVectorizer = WindowsCoOccurrenceVectorizer(self.token2id)

    def process_payload(self, payload: DocumentPayload) -> Any:

        self.token2id = self.pipeline.payload.token2id

        document_id = self.get_document_id(payload)

        tokens: Iterable[str] = payload.content

        if len(tokens) == 0:
            return payload.empty(self.out_content_type)

        self.ingest(tokens)

        windows = tokens_to_windows(tokens=tokens, context_opts=self.context_opts)

        result: WindowsCoOccurrenceOutput = self.vectorizer.fit_transform(windows)

        return payload.update(
            self.out_content_type,
            content=CoOccurrencePayload(
                document_id,
                result.term_term_matrix,
                result.term_window_counter,
            ),
        )

    def get_document_id(self, payload: DocumentPayload) -> int:
        document_id = self.document_index.loc[payload.document_name]['document_id']
        return document_id


@dataclass
class ToCorpusCoOccurrenceDTM(ITask):
    """Computes COMPILED (DOCUMENT-LEVEL) windows co-occurrence data.

    Iterable[DocumentPayload] => ComputeResult
    """

    context_opts: ContextOpts = None
    global_threshold_count: int = 1

    def __post_init__(self):
        self.in_content_type = ContentType.CO_OCCURRENCE_DTM_DOCUMENT
        self.out_content_type = ContentType.CO_OCCURRENCE_DTM_CORPUS

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("context_opts", self.context_opts)
        self.pipeline.put("global_threshold_count", self.global_threshold_count)
        return self

    def process_stream(self) -> VectorizedCorpus:

        if self.document_index is None:
            raise CoOccurrenceError("expected document index found no such thing")

        # FIXME: Do NOT expand stream to list
        stream: Iterable[CoOccurrencePayload] = [payload.content for payload in self.instream if not payload.is_empty]

        # Prevent new tokens from being added
        self.pipeline.payload.token2id.close()

        # FIXME: These test only valid when at least one payload has been processed
        if 'n_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_tokens`, but found no such column")

        if 'n_raw_tokens' not in self.document_index.columns:
            # logger.warning("Value `n_raw_tokens` not in index, using `n_tokens` instead")
            raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no column")

        token2id: Token2Id = self.pipeline.payload.token2id
        document_index: DocumentIndex = self.pipeline.payload.document_index

        corpus: VectorizedCorpus = TTM_to_co_occurrence_DTM(
            stream=stream,
            token2id=token2id,
            document_index=document_index,
        )

        window_counters = itertools.chain((d.document_id, d.term_window_counter) for d in stream)
        window_counts = TokenWindowCountStatistics(
            corpus_counts=self.global_windows_counts(),
            document_counts=self.document_window_counts_matrix(
                window_counters,
                shape=(len(document_index), len(token2id)),
            ),
        )

        yield DocumentPayload(
            content=Bundle(
                corpus=corpus,
                token2id=token2id,
                document_index=document_index,
                window_counts=window_counts,
                compute_options=self.pipeline.payload.stored_opts(),
            )
        )

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None

    def global_windows_counts(self) -> Optional[collections.Counter]:

        task: ToCoOccurrenceDTM = self.pipeline.find(ToCoOccurrenceDTM, self.__class__)
        if task is not None:
            return task.vectorizer.corpus_window_counts
        return task

    def document_window_counts_matrix(
        self, counters: Iterable[Tuple[int, Mapping[int, int]]], shape: tuple
    ) -> sp.spmatrix:
        """Create a matrix with token's window count for each document (rows).
        The shape of the returned sparse matrix is [number of document, vocabulary size]

        Args:
            counters (dict): Dict (key document id) of dict (key token id) of window counts
            shape (tuple): Size of returned sparse matrix

        Returns:
            sp.spmatrix: window counts matrix
        """

        matrix: sp.lil_matrix = sp.lil_matrix(shape, dtype=np.int32)

        for document_id, counts in counters:
            matrix[document_id, list(counts.keys())] = list(counts.values())

        return matrix.tocsr()
