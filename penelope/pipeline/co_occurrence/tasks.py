import sys
from dataclasses import dataclass, field
from pprint import pformat as pf
from typing import Any, Iterable, Mapping, Set, Tuple

from loguru import logger
from penelope.co_occurrence import (
    Bundle,
    ContextOpts,
    CoOccurrenceError,
    TokenWindowCountMatrix,
    VectorizedTTM,
    VectorizeType,
    windows_to_ttm,
)
from penelope.co_occurrence.windows import generate_windows
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.corpus.dtm import WORD_PAIR_DELIMITER
from penelope.pipeline.co_occurrence.tasks_pool import tokens_to_ttm_stream

from ..interfaces import ContentType, DocumentPayload, ITask, PipelineError
from .builder import CoOccurrenceCorpusBuilder, CoOccurrencePayload

sj = WORD_PAIR_DELIMITER.join

DEBUG_TRACE: bool = False

if DEBUG_TRACE:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO", enqueue=True)
    logger.add("co_occurrence_trace.py", rotation=None, format="{message}", serialize=False, level="INFO", enqueue=True)


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

    __concept_ids: Set[int] = field(init=False, default=None)
    __ignore_ids: Set[int] = field(init=False, default=None)

    def __post_init__(self):
        self.in_content_type = [ContentType.TOKENS, ContentType.TOKEN_IDS]
        self.out_content_type = ContentType.CO_OCCURRENCE_DTM_DOCUMENT

    @property
    def concept_ids(self):
        if self.__concept_ids is None:
            self.__concept_ids = {self.pipeline.payload.token2id[t] for t in self.context_opts.get_concepts()}
        return self.__concept_ids

    @property
    def ignore_ids(self):

        if self.__ignore_ids is None:
            self.__ignore_ids = set()

            if self.context_opts.ignore_padding:
                self.__ignore_ids.add(self.pipeline.payload.token2id[self.context_opts.pad])

            if self.context_opts.ignore_concept:
                self.__ignore_ids.update(self.concept_ids)

        return self.__ignore_ids

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("context_opts", self.context_opts)
        return self

    def enter(self):
        super().enter()

        if not self.pipeline.find("Vocabulary", type(self)):
            raise PipelineError(f"{type(self).__name__}: requires preceeding Vocabulary task")

        if self.pipeline.payload.token2id is None:
            raise PipelineError(f"{type(self).__name__} requires a vocabulary!")

        if self.context_opts.pad not in self.pipeline.payload.token2id:
            _ = self.pipeline.payload.token2id[self.context_opts.pad]

    def _process_payload(self, payload: DocumentPayload) -> Any:

        token2id = self.pipeline.payload.token2id
        fg = token2id.data.get

        if len(payload.content) == 0:
            return payload.empty(self.out_content_type)

        token_ids: Iterable[int] = (
            payload.content if self.in_content_type == ContentType.TOKEN_IDS else [fg(t) for t in payload.content]
        )

        windows: Iterable[Iterable[int]] = generate_windows(
            token_ids=token_ids,
            context_width=self.context_opts.context_width,
            pad_id=fg(self.context_opts.pad),
            ignore_pads=self.context_opts.ignore_padding,
        )

        document_id: int = self.get_document_id(payload)
        ttm_map: Mapping[VectorizeType, VectorizedTTM] = windows_to_ttm(
            document_id=document_id,
            windows=windows,
            concept_ids=self.concept_ids,
            ignore_ids=self.ignore_ids,
            vocab_size=len(token2id),
        )

        if DEBUG_TRACE:
            self.trace(payload, windows, token_ids)

        return payload.update(
            self.out_content_type,
            content=CoOccurrencePayload(
                document_id=document_id,
                document_name=payload.document_name,
                ttm_data_map=ttm_map,
            ),
        )

    def process_payload(self, payload: DocumentPayload) -> Any:
        return payload

    def process_stream(self) -> Iterable[DocumentPayload]:
        """Processes stream of payloads. Overridable. """
        # stream: Iterable[Tuple] = self.prepare_task_stream(
        #     token2id=self.pipeline.payload.token2id,
        #     context_opts=self.context_opts,
        # )

        for item in tokens_to_ttm_stream(
            payload_stream=self.prior.outstream(),
            document_index=self.document_index,
            token2id=self.pipeline.payload.token2id,
            context_opts=self.context_opts,
            concept_ids=self.concept_ids,
            ignore_ids=self.ignore_ids,
            processes=self.context_opts.processes,
            chunksize=self.context_opts.chunksize,
        ):
            yield DocumentPayload(
                content_type=self.out_content_type,
                filename=item['filename'],
                content=CoOccurrencePayload(
                    document_id=item.get('document_id'),
                    document_name=item.get('document_name'),
                    ttm_data_map=item.get('ttm_map'),
                ),
            )

    def prepare_task_stream(self, token2id: Token2Id, context_opts: ContextOpts) -> Iterable[Tuple]:

        fg = token2id.data.get
        name_to_id: dict = self.document_index.document_id.to_dict()
        task_stream: Iterable[Tuple] = (
            (
                name_to_id[payload.document_name],
                payload.document_name,
                payload.filename,
                payload.content if payload.content_type == ContentType.TOKEN_IDS else [fg(t) for t in payload.content],
                fg(context_opts.pad),
                context_opts,
                self.concept_ids,
                self.ignore_ids,
                len(token2id),
            )
            for payload in self.prior.outstream()
        )
        return task_stream

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
