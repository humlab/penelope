from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import pandas as pd
from penelope.corpus.readers import TextSource

from ._utils import consolidate_document_index

if TYPE_CHECKING:
    from . import pipeline as corpus_pipeline

# FIXME: #24 Make a GENERIC CORPUS PIPELIME, use MixIn for vendor specific tasks


class ContentType(Enum):
    NONE = 0
    DATAFRAME = 1
    TEXT = 2
    TOKENS = 3
    SPACYDOC = 4
    SPARV_XML = 5
    SPARV_CSV = 6


@dataclass
class DocumentPayload:

    content_type: ContentType = ContentType.NONE
    content: Any = None
    filename: str = None
    filename_values: Mapping[str, Any] = None

    def update(self, content_type: ContentType, content: Any):
        self.content_type = content_type
        self.content = content
        return self


class PipelineError(Exception):
    pass


@dataclass
class PipelinePayload:

    source: TextSource = None
    document_index_filename: str = None
    document_index: pd.DataFrame = None

    memory_store: Mapping[str, Any] = field(default_factory=dict)
    pos_schema_name: str = field(default="Universal")

    # NOT USED: token2id: Mapping = None
    # NOT USED: extract_tokens_opts: ExtractTokensOpts = None
    # NOT USED: tokens_transform_opts: TokensTransformOpts = None
    # NOT USED: extract_opts: Mapping = None

    def get(self, key: str, default=None):
        return self.memory_store(key, default)

    def put(self, key: str, value: Any):
        self.memory_store[key] = value

    def consolidate_document_index(self, reader_index: pd.DataFrame):
        self.document_index = consolidate_document_index(
            self.document_index_filename,
            self.document_index,
            reader_index,
        )
        return self

class ITask(abc.ABC):

    pipeline: corpus_pipeline.CorpusPipeline = None
    instream: Iterable[DocumentPayload] = None

    def chain(self) -> "ITask":
        prior_task = self.pipeline.get_prior_to(self)
        if prior_task is not None:
            self.instream = prior_task.outstream()
        return self

    def setup(self):
        return self

    def outstream(self) -> Iterable[DocumentPayload]:
        for payload in self.instream:
            yield self.process_payload(payload)

    @abc.abstractmethod
    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return payload

    def hookup(self, pipeline: corpus_pipeline.CorpusPipeline) -> corpus_pipeline.CorpusPipeline:
        self.pipeline = pipeline
        return self

    @property
    def document_index(self) -> pd.DataFrame:
        return self.pipeline.payload.document_index
