from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

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
from penelope.corpus.dtm import WORD_PAIR_DELIMITER
from penelope.pipeline.tasks_mixin import VocabularyIngestMixIn
from penelope.type_alias import DocumentIndex

from ..interfaces import ContentType, DocumentPayload, ITask, PipelineError


@dataclass
class CoOccurrencePayload:
    document_id: int
    vectorized_data: Mapping[VectorizeType, VectorizedTTM]


# def to_token_pairs(term_term_matrix: sp.spmatrix, single_vocabulary: Token2Id) -> Iterable[str]:
#     fg = single_vocabulary.id2token.get
#     sep: str = WORD_PAIR_DELIMITER
#     return (
#         f"{fg(a, '').replace(sep, '')}{sep}{fg(b, '').replace(sep, '')}"
#         for (a, b) in zip(term_term_matrix.row, term_term_matrix.col)
#     )
#     # return (to_word_pair_token(a, b, fg) for (a, b) in zip(term_term_matrix.row, term_term_matrix.col))


class CoOccurrenceCorpusBuilder:
    """Creates icrementally a DTM co-occurrence corpus from a stream of document TTM matrices"""

    def __init__(
        self,
        vectorize_type: VectorizeType,
        document_index: DocumentIndex,
        pair_vocabulary: Token2Id,
        single_vocabulary: Token2Id,
    ):
        self.vectorize_type: VectorizeType = vectorize_type
        self.document_index: DocumentIndex = document_index
        self.pair_vocabulary: Token2Id = pair_vocabulary
        self.single_vocabulary: Token2Id = single_vocabulary

        """ Co-occurrence DTM matrix """
        self.matrix: sp.spmatrix = None
        self.row = []
        self.col = []
        self.data = []

        """ Token window counts per document """
        self.counts_row = []
        self.counts_col = []
        self.counts_data = []

        self._single_id2token_without_sep: dict = None

    @property
    def single_id2token_without_sep(self) -> dict:
        if self._single_id2token_without_sep is None:
            self._single_id2token_without_sep = {
                w_id: w.replace(WORD_PAIR_DELIMITER, '') for w_id, w in self.single_vocabulary.id2token.items()
            }
        return self._single_id2token_without_sep

    def ingest(self, payloads: Iterable[CoOccurrencePayload]) -> "CoOccurrenceCorpusBuilder":
        for payload in payloads:
            self.add(payload)
        return self

    def add(self, payload: CoOccurrencePayload) -> None:

        item: VectorizedTTM = payload.vectorized_data.get(self.vectorize_type)

        TTM: scipy.sparse.spmatrix = item.term_term_matrix

        """Translate token-pair ids into id in new COO-vocabulary"""

        # START: OPTIMIZED PERFORMENCE SECTION
        # token_ids = [
        #     self.pair_vocabulary[
        #         sep.join([fg(a, '').replace(sep, ''), fg(b, '').replace(sep, '')])
        #         # f"{fg(a, '').replace(sep, '')}{sep}{fg(b, '').replace(sep, '')}"
        #         # to_word_pair_token(a, b, fg)
        #     ]
        #     for (a, b) in zip(TTM.row, TTM.col)
        # ]

        fg: Callable[[int], str] = self.single_id2token_without_sep.get
        r = TTM.row
        c = TTM.col
        pv = self.pair_vocabulary
        sj = WORD_PAIR_DELIMITER.join
        token_ids = [pv[sj([fg(r[i], ''), fg(c[i], '')])] for i in range(0, len(r))]
        # END: OPTIMIZED PERFORMENCE SECTION

        self.row.extend([item.document_id] * len(token_ids))
        self.col.extend(token_ids)
        self.data.extend(TTM.data)

        """ Add term windows counts """
        counts: Mapping[int, int] = item.term_window_counts
        # self.window_count_matrix[item.document_id, list(counts.keys())] = list(counts.values())
        self.counts_row.extend([item.document_id] * len(counts))
        self.counts_col.extend(counts.keys())
        self.counts_data.extend(counts.values())

    @property
    def corpus(self) -> VectorizedCorpus:
        shape: Tuple[int, int] = (len(self.document_index), len(self.pair_vocabulary))
        self.matrix = sp.coo_matrix((self.data, (self.row, self.col)), shape=shape)
        corpus: VectorizedCorpus = VectorizedCorpus(
            bag_term_matrix=self.matrix.tocsr(),
            token2id=dict(self.pair_vocabulary.data),
            document_index=self.document_index.set_index('document_id', drop=False),
        )

        return corpus

    def compile_window_count_statistics(self, total_term_window_counts: Counter) -> TokenWindowCountStatistics:
        window_count_matrix: sp.spmatrix = sp.coo_matrix((self.counts_data, (self.counts_row, self.counts_col))).tocsr()
        window_counts: TokenWindowCountStatistics = TokenWindowCountStatistics(
            corpus_counts=total_term_window_counts,
            document_counts=window_count_matrix,
        )
        return window_counts

    def ingest_tokens(self, payload: CoOccurrencePayload) -> "CoOccurrenceCorpusBuilder":
        item: VectorizedTTM = payload.vectorized_data.get(self.vectorize_type)
        # START: OPTIMIZED PERFORMENCE SECTION
        # self.pair_vocabulary.ingest(to_token_pairs(item.term_term_matrix, self.single_vocabulary))

        r = item.term_term_matrix.row
        c = item.term_term_matrix.col
        sj = WORD_PAIR_DELIMITER.join
        fg: Callable[[int], str] = self.single_id2token_without_sep.get
        self.pair_vocabulary.ingest(sj([fg(r[i], ''), fg(c[i], '')]) for i in range(0, len(r)))

        # END: OPTIMIZED PERFORMENCE SECTION
        return self


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

        document_id = self.get_document_id(payload)

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

    def process_stream(self) -> Iterable[DocumentPayload]:

        if self.document_index is None:
            raise CoOccurrenceError("expected document index found no such thing")

        """Ingest token-pairs into new COO-vocabulary using existing token vocabulary"""

        single_vocabulary: Token2Id = self.pipeline.payload.token2id
        pair_vocabulary: Token2Id = Token2Id()

        normal_builder: CoOccurrenceCorpusBuilder = CoOccurrenceCorpusBuilder(
            VectorizeType.Normal, self.document_index, pair_vocabulary, single_vocabulary
        )
        concept_builder: CoOccurrenceCorpusBuilder = (
            CoOccurrenceCorpusBuilder(VectorizeType.Concept, self.document_index, pair_vocabulary, single_vocabulary)
            if self.context_opts.concept
            else None
        )

        coo_payloads: Iterable[CoOccurrencePayload] = (
            payload.content
            for payload in self.prior.outstream(desc="Ingest", total=len(self.document_index))
            if payload.content is not None
        )
        for coo_payload in coo_payloads:
            normal_builder.ingest_tokens(coo_payload).add(payload=coo_payload)
            if concept_builder:
                concept_builder.add(payload=coo_payload)

        pair_vocabulary.close()

        payload: DocumentPayload = DocumentPayload(
            content=Bundle(
                corpus=normal_builder.corpus.remember(window_counts=self.get_window_counts(normal_builder)),
                token2id=self.pipeline.payload.token2id,
                document_index=self.pipeline.payload.document_index,
                concept_corpus=concept_builder.corpus.remember(window_counts=self.get_window_counts(concept_builder))
                if concept_builder
                else None,
                compute_options=self.pipeline.payload.stored_opts(),
            )
        )

        yield payload

    def get_window_counts(self, builder: CoOccurrenceCorpusBuilder) -> TokenWindowCountStatistics:
        return (
            builder.compile_window_count_statistics(
                self.vectorizer().total_term_window_counts.get(builder.vectorize_type)
            )
            if builder is not None
            else None
        )

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None

    def vectorizer(self) -> Optional[DocumentWindowsVectorizer]:
        task: ToCoOccurrenceDTM = self.pipeline.find(ToCoOccurrenceDTM, self.__class__)
        if task is not None:
            return task.vectorizer
        return task
