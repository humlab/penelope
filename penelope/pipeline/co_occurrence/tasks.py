import collections
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

import pandas as pd
from penelope.co_occurrence import ContextOpts, CoOccurrenceComputeResult, CoOccurrenceError, partition_by_document
from penelope.corpus import Token2Id, VectorizedCorpus

from ..interfaces import ContentType, DocumentPayload, ITask


@dataclass
class ToDocumentCoOccurrence(ITask):
    """Computes a (DOCUMENT-LEVEL) windows co-occurrence data.

    Iterable[DocumentPayload] => Iterable[DocumentPayload]
    """

    context_opts: ContextOpts = None
    ingest_tokens: bool = True
    vectorizer: partition_by_document.WindowsCoOccurrenceVectorizer = field(init=False, default=None)
    token2id: Token2Id = field(init=False, default=None)

    def __post_init__(self):
        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.CO_OCCURRENCE_DATA_FRAME

    def setup(self) -> ITask:
        super().setup()

        if self.pipeline.payload.token2id is None:
            self.pipeline.payload.token2id = Token2Id().open()

        self.token2id = self.pipeline.payload.token2id
        self.vectorizer: partition_by_document.WindowsCoOccurrenceVectorizer = (
            partition_by_document.WindowsCoOccurrenceVectorizer(self.token2id)
        )

        self.pipeline.put("context_opts", self.context_opts)
        return self

    def process_payload(self, payload: DocumentPayload) -> Any:

        tokens: Iterable[str] = payload.content

        ignore_ids: set = {self.token2id[self.context_opts.pad]} if self.context_opts else None

        if self.ingest_tokens:
            self.token2id.ingest(tokens)

        misses = [x for x in tokens if x not in self.token2id]
        if len(misses) > 0:
            print("MISSES!")
            print(misses)

        co_occurrences: pd.DataFrame = partition_by_document.compute_document_co_occurrence(
            vectorizer=self.vectorizer,
            tokens=tokens,
            context_opts=self.context_opts,
            ignore_ids=ignore_ids,
        )

        co_occurrences['document_id'] = self.get_document_id(payload)

        return payload.update(self.out_content_type, content=co_occurrences)

    def get_document_id(self, payload: DocumentPayload) -> int:
        document_id = self.document_index.loc[payload.document_name]['document_id']
        return document_id


@dataclass
class ToCorpusDocumentCoOccurrence(ITask):
    """Computes a COMPILED (DOCUMENT-LEVEL) windows co-occurrence data.

    Iterable[DocumentPayload] => ComputeResult
    """

    context_opts: ContextOpts = None
    global_threshold_count: int = 1

    def __post_init__(self):
        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.CO_OCCURRENCE_DATA_FRAME

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("context_opts", self.context_opts)
        return self

    def process_stream(self) -> VectorizedCorpus:

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

        task: ToDocumentCoOccurrence = self.pipeline.find(ToDocumentCoOccurrence, self.__class__)
        if task is not None:
            return task.vectorizer.token_windows_counts
        return task

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None


# @deprecated
# @dataclass
# class ToCorpusCoOccurrence(ITask):
#     """Computes a (CORPUS-LEVEL) windows co-occurrence data.

#     Iterable[DocumentPayload] => ComputeResult

#     """

#     context_opts: ContextOpts = None
#     transform_opts: TokensTransformOpts = None
#     global_threshold_count: int = None

#     def __post_init__(self):
#         self.in_content_type = [ContentType.DOCUMENT_CONTENT_TUPLE, ContentType.TOKENS]
#         self.out_content_type = ContentType.CO_OCCURRENCE_DATAFRAME

#     def setup(self) -> ITask:
#         super().setup()
#         self.pipeline.put("context_opts", self.context_opts)
#         self.pipeline.put("global_threshold_count", self.global_threshold_count)
#         return self

#     def process_stream(self) -> VectorizedCorpus:

#         # if self.pipeline.get_prior_content_type(self)  == ContentType.DOCUMENT_CONTENT_TUPLE:
#         instream = (x.content for x in self.instream)
#         # else:
#         #     instream = ((x.filename, x.content) for x in self.instream)

#         value: CoOccurrenceComputeResult = partition_by_key.compute_corpus_co_occurrence(
#             stream=instream,
#             token2id=self.pipeline.payload.token2id,
#             document_index=self.pipeline.payload.document_index,
#             context_opts=self.context_opts,
#             transform_opts=self.transform_opts,
#             global_threshold_count=self.global_threshold_count,
#         )
#         yield DocumentPayload(content_type=ContentType.CO_OCCURRENCE_DATAFRAME, content=value)

#     def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
#         return None
