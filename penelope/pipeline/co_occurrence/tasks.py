import collections
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Callable

import scipy
from penelope.co_occurrence import (
    Bundle,
    ContextOpts,
    CoOccurrenceError,
    WindowsCoOccurrenceVectorizer,
    to_token_window_counts_matrix,
    tokens_to_windows,
)
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.type_alias import DocumentIndex

from ..interfaces import ContentType, DocumentPayload, ITask

CoOccurrencePayload = collections.namedtuple(
    'DocumentCoOccurrenceMatrixBundle', ['document_id', 'term_term_matrix', 'term_windows_count']
)


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

        vocabulary.ingest(f"{fg(a)}/{fg(b)}" for (a, b) in zip(TTM.row, TTM.col))

    vocabulary.close()

    """Create sparse matrix where rows are documents, and columns are "token-pairs" tokens"""
    matrix: scipy.sparse.lil_matrix = scipy.sparse.lil_matrix((len(document_index), len(vocabulary)), dtype=int)

    for item in stream:
        TTM: scipy.sparse.spmatrix = item.term_term_matrix

        """Translate token-pair ids into id in new COO-vocabulary"""
        token_ids = [vocabulary[f"{fg(a)}/{fg(b)}"] for (a, b) in zip(TTM.row, TTM.col)]

        matrix[item.document_id, [token_ids]] = TTM.data

    document_index = document_index.set_index('document_id', drop=False)

    corpus = VectorizedCorpus(bag_term_matrix=matrix.tocsr(), token2id=vocabulary.data, document_index=document_index)

    return corpus


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

        document_id = self.get_document_id(payload)

        tokens: Iterable[str] = payload.content

        if len(tokens) == 0:
            return payload.empty(self.out_content_type)

        if self.ingest_tokens:
            self.token2id.ingest(tokens)

        windows = tokens_to_windows(tokens=tokens, context_opts=self.context_opts)
        windows_ttm_matrix, window_counts = self.vectorizer.fit_transform(windows)

        return payload.update(
            self.out_content_type,
            content=CoOccurrencePayload(
                document_id,
                windows_ttm_matrix,
                window_counts,
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
        return self

    def process_stream(self) -> VectorizedCorpus:

        if self.document_index is None:
            raise CoOccurrenceError("expected document index found no such thingNone")

        # FIXME: Do NOT expand stream to list
        stream: Iterable[CoOccurrencePayload] = [
            payload.content for payload in self.instream if not payload.is_empty
        ]

        # FIXME: These test only valid when at least one payload has been processed
        if 'n_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_tokens`, but found no column")

        if 'n_raw_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no column")

        token2id: Token2Id = self.pipeline.payload.token2id
        document_index: DocumentIndex = self.pipeline.payload.document_index

        corpus: VectorizedCorpus = TTM_to_co_occurrence_DTM(
            stream=stream,
            token2id=token2id,
            document_index=document_index,
        )

        corpus_token_window_counts: collections.Counter = self.get_token_windows_counts()

        document_token_window_counters: dict = {d.document_id: dict(d.term_windows_count) for d in stream}

        document_token_window_count_matrix: scipy.sparse.spmatrix = to_token_window_counts_matrix(document_token_window_counters, corpus.data.shape)

        yield DocumentPayload(
            content=Bundle(
                corpus=corpus,
                token2id=token2id,
                document_index=document_index,
                corpus_token_window_counts=corpus_token_window_counts,
                document_token_window_count_matrix=document_token_window_count_matrix,
            )
        )

    def get_token_windows_counts(self) -> Optional[collections.Counter]:

        task: ToCoOccurrenceDTM = self.pipeline.find(ToCoOccurrenceDTM, self.__class__)
        if task is not None:
            return task.vectorizer.corpus_token_window_counts
        return task

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None
