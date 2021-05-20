import collections
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

import pandas as pd
from penelope.co_occurrence import (
    ContextOpts,
    CoOccurrenceComputeResult,
    CoOccurrenceError,
    WindowsCoOccurrenceVectorizer,
    tokens_to_windows,
)
from penelope.corpus import Token2Id, VectorizedCorpus

from ..interfaces import ContentType, DocumentPayload, ITask

CoOccurrenceMatrixBundle = collections.namedtuple(
    'DocumentCoOccurrenceMatrixBundle', ['document_id', 'term_term_matrix', 'term_windows_count']
)


@dataclass
class ToCoOccurrenceMatrixBundle(ITask):
    """Computes a (DOCUMENT-LEVEL) windows co-occurrence data.

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
        self.out_content_type = ContentType.CO_OCCURRENCE_MATRIX_DOCUMENT_BUNDLE

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
class ToCorpusCoOccurrenceMatrixBundle(ITask):
    """Computes a COMPILED (DOCUMENT-LEVEL) windows co-occurrence data.

    Iterable[DocumentPayload] => ComputeResult
    """

    context_opts: ContextOpts = None
    global_threshold_count: int = 1

    def __post_init__(self):
        self.in_content_type = ContentType.CO_OCCURRENCE_MATRIX_DOCUMENT_BUNDLE
        self.out_content_type = ContentType.CO_OCCURRENCE_MATRIX_CORPUS_BUNDLE

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("context_opts", self.context_opts)
        return self

    def process_stream(self) -> VectorizedCorpus:

        """Merge individual TTM to a single sparse matrix"""

        """Create a sparse matrix [row=document_id, column=token_id, value=count] from document token counts"""

        # for payload in self.instream:
        #     item: CoO
        #     ttm = payload.content.

        total_results: List[pd.DataFrame] = [p.content for p in self.instream]

        co_occurrences: pd.DataFrame = pd.concat(total_results, ignore_index=True)[
            ['document_id', 'w1_id', 'w2_id', 'value']
        ]

        token2id: Token2Id = self.pipeline.payload.token2id

        # FIXME This is already taken care of in ToDocumentCoOccurrence!
        # if self.context_opts.ignore_padding:
        #     pad_id: int = token2id[self.context_opts.pad]
        #     co_occurrences = co_occurrences[((co_occurrences.w1_id != pad_id) & (co_occurrences.w2_id != pad_id))]

        if len(co_occurrences) > 0 and self.global_threshold_count > 1:
            co_occurrences = co_occurrences[
                co_occurrences.groupby(["w1_id", "w2_id"])['value'].transform('sum') >= self.global_threshold_count
            ]

        if self.document_index is None:
            raise CoOccurrenceError("expected document index found None")

        if 'n_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_tokens`, but found no column")

        if 'n_raw_tokens' not in self.document_index.columns:
            raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no column")

        yield DocumentPayload(
            content=CoOccurrenceComputeResult(
                co_occurrences=co_occurrences,
                token2id=token2id,
                document_index=self.document_index,
                token_window_counts=self.get_token_windows_counts(),
            )
        )

    def get_token_windows_counts(self) -> Optional[collections.Counter]:

        task: ToCoOccurrenceMatrixBundle = self.pipeline.find(ToCoOccurrenceMatrixBundle, self.__class__)
        if task is not None:
            return task.vectorizer.global_token_windows_counts
        return task

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None
