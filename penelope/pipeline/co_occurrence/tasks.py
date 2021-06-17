
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import numpy as np
import scipy
import scipy.sparse as sp
from penelope.co_occurrence import (
    Bundle,
    ContextOpts,
    CoOccurrenceError,
    DocumentWindowsVectorizer,
    TokenWindowCountStatistics,
    VectorizeType,
)
from penelope.co_occurrence import VectorizedTTM
from penelope.co_occurrence.windows import generate_windows
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.corpus.dtm import to_word_pair_token
from penelope.pipeline.tasks_mixin import VocabularyIngestMixIn
from penelope.type_alias import DocumentIndex, Token

from ..interfaces import ContentType, DocumentPayload, ITask, PipelineError


@dataclass
class CoOccurrencePayload:
    document_id: int
    vectorized_data: Mapping[VectorizeType, VectorizedTTM]


class CoOccurrenceCorpusBuilder:
    """Creates icrementally a DTM co-occurrence corpus from a stream of document TTM matrices"""

    def __init__(
        self,
        document_index: DocumentIndex,
        pair_vocabulary: Token2Id,
        single_vocabulary: Token2Id,
    ):
        self.document_index: DocumentIndex = document_index
        self.pair_vocabulary: Token2Id = pair_vocabulary
        self.single_vocabulary: Token2Id = single_vocabulary
        self.matrix: scipy.sparse.lil_matrix = scipy.sparse.lil_matrix(
            (len(document_index), len(pair_vocabulary)), dtype=int
        )

    def ingest(self, items: Iterable[VectorizedTTM]) -> "CoOccurrenceCorpusBuilder":
        for item in items:
            self.add(item)
        return self

    def add(self, item: VectorizedTTM) -> None:

        fg: Callable[[int], str] = self.single_vocabulary.id2token.get

        TTM: scipy.sparse.spmatrix = item.term_term_matrix

        """Translate token-pair ids into id in new COO-vocabulary"""
        token_ids = [self.pair_vocabulary[to_word_pair_token(a, b, fg)] for (a, b) in zip(TTM.row, TTM.col)]

        self.matrix[item.document_id, [token_ids]] = TTM.data

    def to_corpus(self) -> VectorizedCorpus:

        corpus: VectorizedCorpus = VectorizedCorpus(
            bag_term_matrix=self.matrix.tocsr(),
            token2id=dict(self.pair_vocabulary.data),
            document_index=self.document_index.set_index('document_id', drop=False),
        )

        return corpus


class DocumentTokenWindowCountsMatrixBuilder:
    """Create a matrix with token's window count for each document (rows).
    The shape of the returned sparse matrix is [number of document, vocabulary size]

    Args:
        counters (dict): Dict (key document id) of dict (key token id) of window counts
        shape (tuple): Size of returned sparse matrix

    Returns:
        sp.spmatrix: window counts matrix
    """

    def __init__(self, shape=Tuple[int, int]):

        self.shape = shape
        self.matrix: sp.lil_matrix = sp.lil_matrix(shape, dtype=np.int32)

    def ingest(self, items: Iterable[VectorizedTTM]) -> "DocumentTokenWindowCountsMatrixBuilder":
        for item in items:
            self.add(item)
        return self

    def add(self, item: VectorizedTTM):  # counts: Mapping[int, int]):
        counts: Mapping[int, int] = item.term_window_counts
        self.matrix[item.document_id, list(counts.keys())] = list(counts.values())

    @property
    def value(self) -> sp.spmatrix:
        return self.matrix.tocsr()


def create_pair_vocabulary(
    stream: Iterable[CoOccurrencePayload],
    single_vocabulary: Token2Id,
) -> Token2Id:
    """Create a new vocabulary for token-pairs in co-occurrence stream"""
    fg: Callable[[int], str] = single_vocabulary.id2token.get
    vocabulary: Token2Id = Token2Id()
    for item in stream:
        TTM: scipy.sparse.spmatrix = item.term_term_matrix
        vocabulary.ingest(to_word_pair_token(a, b, fg) for (a, b) in zip(TTM.row, TTM.col))
    vocabulary.close()
    return vocabulary


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
    vectorizer: DocumentWindowsVectorizer = field(init=False, default=None)

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

        self.vectorizer: DocumentWindowsVectorizer = DocumentWindowsVectorizer(self.token2id)

    def process_payload(self, payload: DocumentPayload) -> Any:

        self.token2id = self.pipeline.payload.token2id

        document_id = self.get_document_id(payload)

        tokens: Iterable[str] = payload.content

        if len(tokens) == 0:
            return payload.empty(self.out_content_type)

        self.ingest(tokens)

        windows: Iterable[Iterable[Token]] = generate_windows(tokens=tokens, context_opts=self.context_opts)

        # FIXME CO-OCCURRENCE VectorizeType
        data: Mapping[VectorizeType, VectorizedTTM] = self.vectorizer.fit_transform(
            document_id=document_id, windows=windows, context_opts=self.context_opts
        )

        return payload.update(
            self.out_content_type,
            content=CoOccurrencePayload(document_id=document_id, vectorized_data=data),
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

        self.pipeline.payload.token2id.close()

        if 'n_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_tokens`, but found no such column")

        if 'n_raw_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no column")

        """Ingest token-pairs into new COO-vocabulary using existing token vocabulary"""

        vectorizer: DocumentWindowsVectorizer = self.vectorizer()
        single_vocabulary: Token2Id = self.pipeline.payload.token2id
        shape: Tuple[int, int] = (len(self.document_index), len(single_vocabulary))

        payloads: Iterable[CoOccurrencePayload] = [payload.content for payload in self.instream if not payload.is_empty]

        pair_vocabulary: Token2Id = Token2Id()

        normal_builder = CoOccurrenceCorpusBuilder(self.document_index, pair_vocabulary, single_vocabulary)
        concept_builder = (
            CoOccurrenceCorpusBuilder(self.document_index, pair_vocabulary, single_vocabulary)
            if self.context_opts.concept
            else None
        )

        normal_counts_builder = DocumentTokenWindowCountsMatrixBuilder(shape=shape)
        concept_counts_builder = (
            DocumentTokenWindowCountsMatrixBuilder(shape=shape) if self.context_opts.concept else None
        )

        for payload in payloads:

            item = payload.vectorized_data.get(VectorizeType.Normal)

            pair_vocabulary.ingest(self.to_token_pairs(item.term_term_matrix, single_vocabulary.id2token.get))

            normal_builder.add(item)
            normal_counts_builder.add(item)

            if concept_builder:
                concept_item = payload.vectorized_data.get(VectorizeType.Concept)
                # pair_vocabulary.ingest(self.to_token_pairs(concept_item.term_term_matrix, single_vocabulary.id2token.get))
                concept_builder.add(concept_item)
                concept_counts_builder.add(item)

        pair_vocabulary.close()

        corpus: VectorizedCorpus = normal_builder.to_corpus()
        concept_corpus: VectorizedCorpus = concept_builder.to_corpus() if concept_builder else None

        normal_window_counts = TokenWindowCountStatistics(
            corpus_counts=vectorizer.total_term_window_counts.get(VectorizeType.Normal),
            document_counts=normal_counts_builder.value,
        )

        concept_window_counts = (
            TokenWindowCountStatistics(
                corpus_counts=vectorizer.total_term_window_counts.get(VectorizeType.Concept),
                document_counts=concept_counts_builder.value,
            )
            if concept_counts_builder
            else None
        )

        yield DocumentPayload(
            content=Bundle(
                corpus=corpus,
                window_counts=normal_window_counts,
                token2id=self.pipeline.payload.token2id,
                document_index=self.pipeline.payload.document_index,
                concept_corpus=concept_corpus,
                concept_window_counts=concept_window_counts,
                compute_options=self.pipeline.payload.stored_opts(),
            )
        )

    def to_token_pairs(self, term_term_matrix: sp.spmatrix, single_vocabulary: Token2Id) -> Iterable[str]:
        fg = single_vocabulary.id2token.get
        return (to_word_pair_token(a, b, fg) for (a, b) in zip(term_term_matrix.row, term_term_matrix.col))

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None

    def vectorizer(self) -> Optional[DocumentWindowsVectorizer]:
        task: ToCoOccurrenceDTM = self.pipeline.find(ToCoOccurrenceDTM, self.__class__)
        if task is not None:
            return task.vectorizer
        return task
