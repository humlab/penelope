from __future__ import annotations

import abc
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence, Tuple, Union

import pandas as pd
from penelope.corpus.readers import TextSource
from penelope.utility.filename_utils import replace_extension

from ._utils import load_document_index, read_data_frame_from_zip, write_data_frame_to_zip

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
    VECTORIZED_CORPUS = 6


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

    def as_str(self):
        if self.content_type == ContentType.TEXT:
            return self.content
        if self.content_type == ContentType.TOKENS:
            return ' '.join(self.content)
        raise PipelineError(f"payload of content type {self.content_type} cannot be stringified")


class PipelineError(Exception):
    pass


@dataclass
class PipelinePayload:

    source: TextSource = None
    document_index_source: Union[str, pd.DataFrame] = None

    _document_index: pd.DataFrame = None

    memory_store: Mapping[str, Any] = field(default_factory=dict)
    pos_schema_name: str = field(default="Universal")

    # NOT USED: token2id: Mapping = None
    # NOT USED: extract_tokens_opts: ExtractTokensOpts = None
    # NOT USED: tokens_transform_opts: TokensTransformOpts = None
    # NOT USED: extract_opts: Mapping = None

    def get(self, key: str, default=None):
        return self.memory_store.get(key, default)

    def put(self, key: str, value: Any):
        self.memory_store[key] = value

    @property
    def document_index(self) -> pd.DataFrame:

        if self._document_index is None:
            if self.document_index_source is not None:
                if isinstance(self.document_index_source, pd.DataFrame):
                    self._document_index = self.document_index_source
                elif isinstance(self.document_index_source, str):
                    self._document_index = load_document_index(self.document_index_source)

        return self._document_index

    @document_index.setter
    def document_index(self, value: pd.DataFrame):

        self.document_index_source = value
        self._document_index = value


@dataclass
class ITask(abc.ABC):

    pipeline: corpus_pipeline.CorpusPipeline = None
    instream: Iterable[DocumentPayload] = None

    in_content_type: Union[ContentType, Sequence[ContentType]] = field(init=False, default=None)
    out_content_type: ContentType = field(init=False, default=None)

    def chain(self) -> "ITask":
        prior_task = self.pipeline.get_prior_to(self)
        if prior_task is not None:
            self.instream = prior_task.outstream()
        return self

    def setup(self):
        return self

    @abc.abstractmethod
    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return payload

    def process(self, payload: DocumentPayload) -> DocumentPayload:
        self.input_type_guard(payload.content_type)
        return self.process_payload(payload)

    def outstream(self) -> Iterable[DocumentPayload]:
        if self.instream is None:
            raise PipelineError("No instream specified. Have you loaded a corpus source?")
        for payload in self.instream:
            yield self.process(payload)

    def hookup(self, pipeline: corpus_pipeline.CorpusPipeline) -> corpus_pipeline.CorpusPipeline:
        self.pipeline = pipeline
        return self

    @property
    def document_index(self) -> pd.DataFrame:
        return self.pipeline.payload.document_index

    def input_type_guard(self, content_type):
        if self.in_content_type is None:
            return
        if isinstance(self.in_content_type, ContentType):
            if self.in_content_type == content_type:
                return
        if isinstance(
            self.in_content_type,
            (
                list,
                tuple,
            ),
        ):
            if content_type in self.in_content_type:
                return
        raise PipelineError("content type not valid for task")
