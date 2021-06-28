import sys
from dataclasses import dataclass, field
from pprint import pformat as pf
from typing import Any, Iterable, Mapping, Optional, Tuple

import scipy
import scipy.sparse as sp
from loguru import logger
from penelope.co_occurrence import (
    Bundle,
    ContextOpts,
    CoOccurrenceError,
    DocumentWindowsVectorizer,
    TokenWindowCountMatrix,
    VectorizedTTM,
    VectorizeType,
)
from penelope.co_occurrence.windows import generate_windows
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.corpus.dtm import WORD_PAIR_DELIMITER
from penelope.pipeline.tasks_mixin import VocabularyIngestMixIn
from penelope.type_alias import DocumentIndex

from ..interfaces import ContentType, DocumentPayload, ITask, PipelineError

sj = WORD_PAIR_DELIMITER.join

DEBUG_TRACE: bool = False
if DEBUG_TRACE:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO", enqueue=True)
    logger.add("co_occurrence_trace.py", rotation=None, format="{message}", serialize=False, level="INFO", enqueue=True)


@dataclass
class CoOccurrencePayload:
    document_id: int
    document_name: str
    ttm_data_map: Mapping[VectorizeType, VectorizedTTM]


class CoOccurrenceCorpusBuilder:
    """Creates icrementally a DTM co-occurrence corpus from a stream of document TTM matrices"""

    def __init__(
        self,
        vectorize_type: VectorizeType,
        document_index: DocumentIndex,
        pair2id: Token2Id,
        token2id: Token2Id,
    ):
        self.vectorize_type: VectorizeType = vectorize_type
        self.document_index: DocumentIndex = document_index
        self.pair2id: Token2Id = pair2id
        self.token2id: Token2Id = token2id

        """ Co-occurrence DTM matrix """
        self.matrix: sp.spmatrix = None
        self.row = []
        self.col = []
        self.data = []

        """ Per document term window counts """
        self.dtw_counts_row = []
        self.dtw_counts_col = []
        self.dtw_counts_data = []

    # def ingest(self, payloads: Iterable[CoOccurrencePayload]) -> "CoOccurrenceCorpusBuilder":
    #     for payload in payloads:
    #         self.add(payload)
    #     return self

    def add(self, payload: CoOccurrencePayload) -> None:
        """Adds payload to the DTM-data under construction.
        Note! Assumes that data has been ingested to both vocabulary and vocabs mapping."""

        item: VectorizedTTM = payload.ttm_data_map.get(self.vectorize_type)

        if DEBUG_TRACE:
            item.trace()

        pair2id: Token2Id = self.pair2id

        """Translate token-pair ids into new COO-vocabulary ids"""
        TTM: scipy.sparse.spmatrix = item.term_term_matrix
        pair_ids = (pair2id[p] for p in zip(TTM.row, TTM.col))

        self.row.extend([item.document_id] * len(TTM.row))
        self.col.extend(pair_ids)
        self.data.extend(TTM.data)

        """ Add term windows counts """
        counts: Mapping[int, int] = item.term_window_counts

        self.dtw_counts_row.extend([item.document_id] * len(counts))
        self.dtw_counts_col.extend(counts.keys())
        self.dtw_counts_data.extend(counts.values())

        if DEBUG_TRACE:
            self.trace(f"Document ID {item.document_id}")

    @property
    def corpus(self) -> VectorizedCorpus:
        shape: Tuple[int, int] = (len(self.document_index), len(self.pair2id))
        self.matrix = sp.coo_matrix((self.data, (self.row, self.col)), shape=shape)
        corpus: VectorizedCorpus = VectorizedCorpus(
            bag_term_matrix=self.matrix.tocsr(),
            token2id=dict(self.pair2id.data),
            document_index=self.document_index.set_index('document_id', drop=False),
        )

        return corpus

    def compile_window_count_matrix(self) -> TokenWindowCountMatrix:
        window_count_matrix: sp.spmatrix = sp.coo_matrix(
            (self.dtw_counts_data, (self.dtw_counts_row, self.dtw_counts_col))
        ).tocsr()
        matrix: TokenWindowCountMatrix = TokenWindowCountMatrix(document_term_window_counts=window_count_matrix)
        return matrix

    def ingest_pairs(self, payload: CoOccurrencePayload) -> "CoOccurrenceCorpusBuilder":
        """Ingests tokens into pair-vocabulary.
        Note: Tokens are at this stage ingested as `integer tuple pairs` i.e. (w1_id, w2_id) instead of tokens string
              The integer pairs are later updated to `w1/w2` string tokens.
              In this way the pair vocabulary keeps a (w1_id, w2_id) to pair_id mapping
        """

        ttm_item: VectorizedTTM = payload.ttm_data_map.get(self.vectorize_type)

        pairs = list(zip(ttm_item.term_term_matrix.row, ttm_item.term_term_matrix.col))

        self.pair2id.ingest(pairs)

        return self

    def trace(self, msg: str) -> None:
        logger.info(
            "\n#################################################################################################"
        )
        logger.info("# CoOccurrenceCorpusBuilder")
        logger.info(f"# VectorizeType.{self.vectorize_type.name}")
        logger.info(f"# {msg}")
        logger.info(
            "#################################################################################################\n"
        )
        logger.info(f"vectorize_type = VectorizeType.{self.vectorize_type.name}")
        # logger.info(f"co_occurrence_dtm_matrix: {pf(list(zip(self.row, self.col, self.data)))}")
        # logger.info(
        #     f"document_term_windows_counts = {pf(list(zip(self.dtw_counts_row, self.dtw_counts_col, self.dtw_counts_data)))}"
        # )
        logger.info(f"token2id = {pf(dict(self.token2id.data), compact=True, width=200)}")
        logger.info(f"pair2id = {pf(dict(self.pair2id.data), compact=True, width=1000)}")
        logger.info(f"co_occurrence_corpus = {pf(self.corpus.data.todense())}")
        shape = (len(self.document_index), len(self.token2id))
        logger.info(
            f"document_term_windows_counts = {pf( sp.coo_matrix((self.dtw_counts_data, (self.dtw_counts_row, self.dtw_counts_col)), shape=shape).todense())}"
        )


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
        self.in_content_type = [ContentType.TOKENS, ContentType.TOKEN_IDS]
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
        fg = self.token2id.data.get

        document_id: int = self.get_document_id(payload)

        if len(payload.content) == 0:
            return payload.empty(self.out_content_type)

        if self.in_content_type == ContentType.TOKEN_IDS:
            token_ids: Iterable[int] = payload.content
        else:
            if self.ingest_tokens and self.token2id.is_open:
                # FIXME: Make a version of ingest that returns ids
                self.token2id.ingest(payload.content)
            token_ids: Iterable[int] = [fg(t) for t in payload.content]

        windows: Iterable[Iterable[int]] = generate_windows(
            token_ids=token_ids,
            context_width=self.context_opts.context_width,
            pad_id=fg(self.context_opts.pad),
        )

        windows = list(windows)

        ttm_map: Mapping[VectorizeType, VectorizedTTM] = self.vectorizer.fit_transform(
            document_id=document_id, windows=windows, context_opts=self.context_opts
        )

        if DEBUG_TRACE:
            self.trace(payload, windows, token_ids)

        return payload.update(
            self.out_content_type,
            content=CoOccurrencePayload(
                document_id=document_id, document_name=payload.document_name, ttm_data_map=ttm_map
            ),
        )

    def get_document_id(self, payload: DocumentPayload) -> int:
        document_id = self.document_index.loc[payload.document_name]['document_id']
        return document_id

    def trace(self, payload: DocumentPayload, windows: Iterable[Iterable[int]], token_ids: Iterable[int]) -> None:
        logger.info(
            "\n#################################################################################################"
        )
        logger.info(f"# ToCoOccurrence {payload.document_name}")
        logger.info(
            "#################################################################################################\n"
        )
        logger.info(f"document_name = '{payload.document_name}'")
        logger.info(f"document_id = {self.get_document_id(payload)}")
        logger.info(f"tokens = {pf(payload.content)}")
        logger.info(f"token_ids = {pf(token_ids)}")
        logger.info(f"windows = {pf(windows)}")
        logger.info(f"context_opts = {pf(self.context_opts, compact=True)}")


@dataclass
class ToCorpusCoOccurrenceDTM(ITask):
    """Computes COMPILED (DOCUMENT-LEVEL) windows co-occurrence data.

    Iterable[DocumentPayload] => ComputeResult
    """

    context_opts: ContextOpts = None
    global_threshold_count: int = 1
    compress: bool = False

    def __post_init__(self):
        self.in_content_type = ContentType.CO_OCCURRENCE_DTM_DOCUMENT
        self.out_content_type = ContentType.CO_OCCURRENCE_DTM_CORPUS

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("context_opts", self.context_opts)
        self.pipeline.put("global_threshold_count", self.global_threshold_count)
        return self

    def process_stream(self) -> Iterable[DocumentPayload]:

        if self.document_index is None:
            raise CoOccurrenceError("expected document index found no such thing")

        token2id: Token2Id = self.pipeline.payload.token2id
        pair2id: Token2Id = Token2Id()

        normal_builder: CoOccurrenceCorpusBuilder = CoOccurrenceCorpusBuilder(
            VectorizeType.Normal, self.document_index, pair2id, token2id
        )

        concept_builder: CoOccurrenceCorpusBuilder = (
            CoOccurrenceCorpusBuilder(VectorizeType.Concept, self.document_index, pair2id, token2id)
            if self.context_opts.concept
            else None
        )

        coo_payloads: Iterable[CoOccurrencePayload] = (
            payload.content
            for payload in self.prior.outstream(desc="Ingest", total=len(self.document_index))
            if payload.content is not None
        )
        for coo_payload in coo_payloads:
            normal_builder.ingest_pairs(coo_payload).add(payload=coo_payload)
            if concept_builder:
                concept_builder.add(payload=coo_payload)

        pair2id.close()

        """Translation between id-pair (single vocab IDs) and pair-pid (pair vocab IDs)"""
        token_ids_2_pair_id: Mapping[Tuple[int, int], int] = dict(pair2id.data)

        self.translate_id_pair_to_token(pair2id, token2id)

        concept_corpus: VectorizedCorpus = (
            concept_builder.corpus.remember(window_counts=self.get_window_counts(concept_builder))
            if concept_builder
            else None
        )

        corpus: VectorizedCorpus = normal_builder.corpus.remember(window_counts=self.get_window_counts(normal_builder))

        bundle: Bundle = Bundle(
            corpus=corpus,
            token2id=token2id,
            document_index=self.pipeline.payload.document_index,
            concept_corpus=concept_corpus,
            compute_options=self.pipeline.payload.stored_opts(),
            vocabs_mapping=token_ids_2_pair_id,
        )

        if self.compress:
            bundle.compress()

        payload: DocumentPayload = DocumentPayload(content=bundle)

        yield payload

    def translate_id_pair_to_token(self, pair2id: Token2Id, token2id: Token2Id) -> None:
        """Translates `id pairs` (w1_id, w2_id) to pair-token `w1/w2`"""
        _single_without_sep = {w_id: w.replace(WORD_PAIR_DELIMITER, '') for w_id, w in token2id.id2token.items()}
        sg = _single_without_sep.get
        pair2id.replace(data={sj([sg(w1_id), sg(w2_id)]): pair_id for (w1_id, w2_id), pair_id in pair2id.data.items()})

    def get_window_counts(self, builder: CoOccurrenceCorpusBuilder) -> TokenWindowCountMatrix:
        return builder.compile_window_count_matrix() if builder is not None else None

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None

    def vectorizer(self) -> Optional[DocumentWindowsVectorizer]:
        task: ToCoOccurrenceDTM = self.pipeline.find(ToCoOccurrenceDTM, self.__class__)
        if task is not None:
            return task.vectorizer
        return task

    def trace(self, payload: DocumentPayload, windows: Iterable[Iterable[int]], token_ids: Iterable[int]) -> None:
        logger.info(
            "\n#################################################################################################"
        )
        logger.info(f"# ToCoOccurrence {payload.document_name}")
        logger.info(
            "#################################################################################################\n"
        )
        logger.info(f"document_name = '{payload.document_name}'")
        logger.info(f"tokens = {pf(payload.content)}")
        logger.info(f"token_ids = {pf(token_ids)}")
        logger.info(f"windows = {pf(windows)}")
        logger.info(f"context_opts = {pf(self.context_opts, compact=True)}")
