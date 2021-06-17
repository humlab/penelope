import itertools
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
    VectorizedTTM,
    VectorizeType,
)
from penelope.co_occurrence.windows import generate_windows
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.corpus.dtm import to_word_pair_token
from penelope.pipeline.tasks_mixin import VocabularyIngestMixIn
from penelope.type_alias import DocumentIndex, Token
from tqdm import tqdm

from ..interfaces import ContentType, DocumentPayload, ITask, PipelineError


@dataclass
class CoOccurrencePayload:
    document_id: int
    vectorized_data: Mapping[VectorizeType, VectorizedTTM]


class CoOccurrenceCorpusBuilder:
    """Creates a DTM-corpus from a sequence if document TTM matrices"""

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

    def ingest(self, items: Iterable[CoOccurrencePayload]) -> None:
        for item in items:
            self.add(item)

    def add(self, item: CoOccurrencePayload) -> None:

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


# def TTM_to_co_occurrence_DTM(
#     stream: Iterable[CoOccurrencePayload],
#     document_index: DocumentIndex,
#     pair_vocabulary: Token2Id,
#     single_vocabulary: Token2Id,
# ) -> VectorizedCorpus:
#     """Tranforms a sequence of document-wise term-term matrices to a corpus-wide document-term matrix"""

#     """NOTE: This implementation depends on stream being reiterable..."""
#     fg: Callable[[int], str] = single_vocabulary.id2token.get

#     """Create sparse matrix where rows are documents, and columns are "token-pairs" tokens"""
#     matrix: scipy.sparse.lil_matrix = scipy.sparse.lil_matrix((len(document_index), len(pair_vocabulary)), dtype=int)
#     for item in stream:

#         TTM: scipy.sparse.spmatrix = item.term_term_matrix

#         """Translate token-pair ids into id in new COO-vocabulary"""
#         token_ids = [pair_vocabulary[to_word_pair_token(a, b, fg)] for (a, b) in zip(TTM.row, TTM.col)]

#         matrix[item.document_id, [token_ids]] = TTM.data

#     document_index = document_index.set_index('document_id', drop=False)

#     corpus = VectorizedCorpus(
#         bag_term_matrix=matrix.tocsr(), token2id=dict(pair_vocabulary.data), document_index=document_index
#     )

#     return corpus


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

        # FIXME: Do NOT expand stream to list
        payloads: Iterable[CoOccurrencePayload] = [payload.content for payload in self.instream if not payload.is_empty]

        # Prevent new tokens from being added
        self.pipeline.payload.token2id.close()

        if 'n_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_tokens`, but found no such column")

        if 'n_raw_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no column")

        """Ingest token-pairs into new COO-vocabulary using existing token vocabulary"""

        vectorizer: DocumentWindowsVectorizer = self.vectorizer()

        single_vocabulary: Token2Id = self.pipeline.payload.token2id

        pair_vocabulary: Token2Id = self.create_pair_vocabulary(payloads, single_vocabulary)

        """Create co-occurrence corpus for entire corpus"""
        corpus, window_counts = self.create_corpus(
            payloads=payloads,
            vectorize_type=VectorizeType.Normal,
            pair_vocabulary=pair_vocabulary,
            single_vocabulary=single_vocabulary,
            vectorizer=vectorizer,
        )

        """Create co-occurrence corpus concept windows"""
        concept_corpus, concept_window_counts = (
            self.create_corpus(
                payloads=payloads,
                vectorize_type=VectorizeType.Concept,
                pair_vocabulary=pair_vocabulary,
                single_vocabulary=single_vocabulary,
                vectorizer=vectorizer,
            )
            if self.context_opts.concept
            else (None, None)
        )

        yield DocumentPayload(
            content=Bundle(
                corpus=corpus,
                window_counts=window_counts,
                token2id=self.pipeline.payload.token2id,
                document_index=self.pipeline.payload.document_index,
                concept_corpus=concept_corpus,
                concept_window_counts=concept_window_counts,
                compute_options=self.pipeline.payload.stored_opts(),
            )
        )

    def create_pair_vocabulary(
        self, stream: Iterable[CoOccurrencePayload], single_vocabulary: Token2Id, progress: bool = True
    ) -> Token2Id:

        ttm_stream: Iterable[VectorizedTTM] = (x.vectorized_data.get(VectorizeType.Normal) for x in stream)

        if progress:
            ttm_stream = tqdm(ttm_stream, desc="Vocab (word-pair)", total=len(self.document_index))

        pair_vocabulary: Token2Id = create_pair_vocabulary(stream=ttm_stream, single_vocabulary=single_vocabulary)

        return pair_vocabulary

    # FIXME: Make incremental, addative
    def create_corpus(
        self,
        *,
        payloads: Iterable[CoOccurrencePayload],
        vectorize_type: VectorizeType,
        pair_vocabulary: Token2Id,
        single_vocabulary: Token2Id,
        vectorizer: DocumentWindowsVectorizer,
        progress: bool = True,
    ) -> Tuple[VectorizedCorpus, TokenWindowCountStatistics]:

        ttm_stream: Iterable[VectorizedTTM] = (x.vectorized_data.get(vectorize_type) for x in payloads)

        if progress:
            ttm_stream = tqdm(ttm_stream, desc=f"Corpus ({vectorize_type.name})", total=len(self.document_index))

        corpus: VectorizedCorpus = (
            CoOccurrenceCorpusBuilder(
                single_vocabulary=single_vocabulary,
                pair_vocabulary=pair_vocabulary,
                document_index=self.pipeline.payload.document_index,
            )
            .ingest(ttm_stream)
            .to_corpus()
        )

        # corpus: VectorizedCorpus = TTM_to_co_occurrence_DTM(
        #     stream=ttm_stream,
        #     single_vocabulary=single_vocabulary,
        #     pair_vocabulary=pair_vocabulary,
        #     document_index=self.pipeline.payload.document_index,
        # )

        total_windows_counts: Counter = vectorizer.total_term_window_counts.get(vectorize_type)

        window_counters = itertools.chain((d.document_id, d.term_window_counts) for d in ttm_stream)
        window_counts = TokenWindowCountStatistics(
            corpus_counts=total_windows_counts,
            document_counts=self.to_document_window_counts_matrix(
                window_counters,
                shape=(len(self.pipeline.payload.document_index), len(self.pipeline.payload.token2id)),
            ),
        )

        return corpus, window_counts

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None

    def vectorizer(self) -> Optional[DocumentWindowsVectorizer]:
        task: ToCoOccurrenceDTM = self.pipeline.find(ToCoOccurrenceDTM, self.__class__)
        if task is not None:
            return task.vectorizer
        return task

    def to_document_window_counts_matrix(
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
