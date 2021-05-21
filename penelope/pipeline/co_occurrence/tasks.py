import collections
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
import scipy
from penelope.co_occurrence import (
    Bundle,
    ContextOpts,
    CoOccurrenceError,
    WindowsCoOccurrenceVectorizer,
    tokens_to_windows,
)
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.type_alias import DocumentIndex

from ..interfaces import ContentType, DocumentPayload, ITask

CoOccurrenceMatrixBundle = collections.namedtuple(
    'DocumentCoOccurrenceMatrixBundle', ['document_id', 'term_term_matrix', 'term_windows_count']
)


def TTM_to_coo_DTM(
    stream: Iterable[CoOccurrenceMatrixBundle], token2id: Token2Id, document_index: DocumentIndex
) -> VectorizedCorpus:

    """Note: vocab size must be known..."""

    """Compute vocab size: Number of elements in upper triangular TTM, diagonal excluded"""
    N = len(token2id)  # size of TTM is N x N
    vocab_size = int(N * (N - 1) / 2)

    """Create COO-DTM matrix"""
    shape = (len(document_index), vocab_size)
    matrix: scipy.sparse.lil_matrix = scipy.sparse.lil_matrix(shape, dtype=int)

    """Ingest token-pairs into new vocabulary"""
    fg = token2id.id2token.get
    token2id: Token2Id = Token2Id()

    for item in stream:

        TTM: scipy.sparse.spmatrix = item.term_term_matrix

        """Ingest token-pairs into the new COO-vocabulary"""
        token2id.ingest(f"{fg(a)}/{fg(b)}" for (a, b) in zip(TTM.row, TTM.col))

        """Translate token-pair ids into id in new COO-vocabulary"""
        token_ids = [token2id[f"{fg(a)}/{fg(b)}"] for (a, b) in zip(TTM.row, TTM.col)]

        matrix[item.document_id, [token_ids]] = TTM.data

    document_index = document_index.set_index('document_id', drop=False)

    corpus = VectorizedCorpus(bag_term_matrix=matrix.tocsr(), token2id=token2id.data, document_index=document_index)

    return corpus


def create_document_token_window_counts_matrix(stream: Iterable[CoOccurrenceMatrixBundle], shape: tuple):

    counters: dict = {d.document_id: dict(d.term_windows_count) for d in stream}

    matrix: scipy.sparse.lil_matrix = scipy.sparse.lil_matrix(shape, dtype=np.uint16)

    for document_id, counts in counters.items():
        matrix[document_id, list(counts.keys())] = list(counts.values())

    return matrix.tocsr()


@dataclass
class ToCoOccurrenceDTM(ITask):
    """Computes (DOCUMENT-LEVEL) windows co-occurrence.

    Bundle consists of the following document level information:

        1) Co-occurrence matrix (TTM) with number of common windows in document
        2) Mapping with number of windows each term occurs in

    Iterable[DocumentPayload] => Iterable[DocumentPayload]
        DocumentPayload.content = Tuple[document_id, TTM, token_window_counts]

    """

    context_opts: ContextOpts = None
    ingest_tokens: bool = True
    vectorizer: WindowsCoOccurrenceVectorizer = field(init=False, default=None)
    token2id: Token2Id = field(init=False, default=None)

    def __post_init__(self):
        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.CO_OCCURRENCE_DTM_DOCUMENT

    def setup(self) -> ITask:
        super().setup()

        if self.pipeline.payload.token2id is None:
            self.pipeline.payload.token2id = Token2Id().open()

        self.token2id = self.pipeline.payload.token2id
        self.vectorizer: WindowsCoOccurrenceVectorizer = WindowsCoOccurrenceVectorizer(self.token2id)

        self.pipeline.put("context_opts", self.context_opts)
        return self

    def process_payload(self, payload: DocumentPayload) -> Any:

        tokens: Iterable[str] = payload.content

        if self.ingest_tokens:
            self.token2id.ingest(tokens)

        document_id = self.get_document_id(payload)

        windows = tokens_to_windows(tokens=tokens, context_opts=self.context_opts)
        windows_ttm_matrix: VectorizedCorpus = self.vectorizer.fit_transform(windows)
        token_window_counts: collections.Counter = self.vectorizer.token_window_counts

        return payload.update(
            self.out_content_type,
            content=CoOccurrenceMatrixBundle(document_id, windows_ttm_matrix, token_window_counts),
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
        return self

    def process_stream(self) -> VectorizedCorpus:

        if self.document_index is None:
            raise CoOccurrenceError("expected document index found None")

        if 'n_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_tokens`, but found no column")

        if 'n_raw_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no column")

        # FIXME: Do NOT expand stream to list
        stream: Iterable[CoOccurrenceMatrixBundle] = [payload.content for payload in self.instream]
        token2id: Token2Id = self.pipeline.payload.token2id
        document_index: DocumentIndex = self.pipeline.payload.document_index

        corpus: VectorizedCorpus = TTM_to_coo_DTM(
            stream=stream,
            token2id=token2id,
            document_index=document_index,
        )

        window_counts_global: collections.Counter = self.get_token_windows_counts()
        window_counts_document = create_document_token_window_counts_matrix(stream, corpus.data.shape)

        yield DocumentPayload(
            content=Bundle(
                corpus=corpus,
                token2id=token2id,
                document_index=document_index,
                window_counts_global=window_counts_global,
                window_counts_document=window_counts_document,
            )
        )

    def get_token_windows_counts(self) -> Optional[collections.Counter]:

        task: ToCoOccurrenceDTM = self.pipeline.find(ToCoOccurrenceDTM, self.__class__)
        if task is not None:
            return task.vectorizer.window_counts_global
        return task

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None
