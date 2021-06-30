# from dataclasses import dataclass, field
# from typing import Any, Iterable, List, Mapping

# import pandas as pd
# from penelope.co_occurrence import (
#     Bundle,
#     ContextOpts,
#     CoOccurrenceError,
#     DocumentWindowsVectorizer,
#     TokenWindowCountMatrix,
#     term_term_matrix_to_co_occurrences,
# )
# from penelope.co_occurrence.vectorize import VectorizedTTM, VectorizeType
# from penelope.co_occurrence.windows import tokens_to_windows_
# from penelope.corpus import Token2Id, VectorizedCorpus

# from ...interfaces import ContentType, DocumentPayload, ITask


# @dataclass
# class ToDocumentCoOccurrence(ITask):
#     """Computes a (DOCUMENT-LEVEL) windows co-occurrence data.

#     Iterable[DocumentPayload] => Iterable[DocumentPayload]
#     """

#     context_opts: ContextOpts = None
#     ingest_tokens: bool = True
#     vectorizer: DocumentWindowsVectorizer = field(init=False, default=None)
#     token2id: Token2Id = field(init=False, default=None)

#     def __post_init__(self):
#         self.in_content_type = ContentType.TOKENS
#         self.out_content_type = ContentType.CO_OCCURRENCE_DATA_FRAME_LEGACY

#     def setup(self) -> ITask:
#         super().setup()

#         if self.pipeline.payload.token2id is None:
#             self.pipeline.payload.token2id = Token2Id().open()

#         self.token2id = self.pipeline.payload.token2id
#         self.vectorizer: DocumentWindowsVectorizer = DocumentWindowsVectorizer(self.token2id)

#         self.pipeline.put("context_opts", self.context_opts)

#         return self

#     def process_payload(self, payload: DocumentPayload) -> Any:

#         if not self.context_opts.concept:
#             raise ValueError("ToDocumentCoOccurrence (legacy) only applicable when concept is specified")

#         tokens: Iterable[str] = payload.content

#         ignore_ids: set = {self.token2id[self.context_opts.pad]} if self.context_opts else None

#         if self.ingest_tokens:
#             self.token2id.ingest(tokens)

#         windows = tokens_to_windows_(tokens=tokens, context_opts=self.context_opts)

#         # FIXME CO-OCCURRENCE VectorizeType
#         data: Mapping[VectorizeType, VectorizedTTM] = self.vectorizer.fit_transform(
#             document_id=payload.document_id, windows=windows, context_opts=self.context_opts
#         )

#         co_occurrences = term_term_matrix_to_co_occurrences(
#             data.get(VectorizeType.Concept).windows_ttm_matrix,
#             threshold_count=1,
#             ignore_ids=ignore_ids,
#         )

#         co_occurrences['document_id'] = self.get_document_id(payload)

#         return payload.update(self.out_content_type, content=co_occurrences)

#     def get_document_id(self, payload: DocumentPayload) -> int:
#         document_id = self.document_index.loc[payload.document_name]['document_id']
#         return document_id


# @dataclass
# class ToCorpusDocumentCoOccurrence(ITask):
#     """Computes a COMPILED (DOCUMENT-LEVEL) windows co-occurrence data.

#     Iterable[DocumentPayload] => ComputeResult
#     """

#     context_opts: ContextOpts = None
#     global_threshold_count: int = 1

#     def __post_init__(self):
#         self.in_content_type = ContentType.TOKENS
#         self.out_content_type = ContentType.CO_OCCURRENCE_DATA_FRAME_LEGACY

#     def setup(self) -> ITask:
#         super().setup()
#         self.pipeline.put("context_opts", self.context_opts)
#         return self

#     def process_stream(self) -> VectorizedCorpus:

#         total_results: List[pd.DataFrame] = [p.content for p in self.create_instream()]

#         co_occurrences: pd.DataFrame = pd.concat(total_results, ignore_index=True)[
#             ['document_id', 'w1_id', 'w2_id', 'value']
#         ]

#         token2id: Token2Id = self.pipeline.payload.token2id

#         if len(co_occurrences) > 0 and self.global_threshold_count > 1:
#             co_occurrences = co_occurrences[
#                 co_occurrences.groupby(["w1_id", "w2_id"])['value'].transform('sum') >= self.global_threshold_count
#             ]

#         if self.document_index is None:
#             raise CoOccurrenceError("expected document index found None")

#         if 'n_tokens' not in self.document_index.columns:
#             raise CoOccurrenceError("expected `document_index.n_tokens`, but found no column")

#         if 'n_raw_tokens' not in self.document_index.columns:
#             raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no column")

#         corpus = VectorizedCorpus.from_co_occurrences(
#             co_occurrences=co_occurrences, token2id=token2id, document_index=self.document_index
#         )

#         yield DocumentPayload(
#             content=Bundle(
#                 corpus=corpus.remember(window_counts=TokenWindowCountMatrix()),
#                 co_occurrences=co_occurrences,
#                 token2id=token2id,
#                 document_index=self.document_index,
#             )
#         )

#     def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
#         return None
