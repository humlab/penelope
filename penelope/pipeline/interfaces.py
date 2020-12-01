from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence, Union

import pandas as pd
from penelope.corpus import consolidate_document_index, load_document_index
from penelope.corpus.readers import TextSource

if TYPE_CHECKING:
    from . import pipelines


@unique
class ContentType(IntEnum):
    NONE = 0
    TAGGEDFRAME = 1
    TEXT = 2
    TOKENS = 3
    SPACYDOC = 4
    SPARV_XML = 5
    SPARV_CSV = 6
    TOKENIZED_CORPUS = 7
    VECTORIZED_CORPUS = 8
    # DTM = 9
    # BOW = 10
    ANY = 11
    PASSTHROUGH = 12
    DOCUMENT_CONTENT_TUPLE = 13


@dataclass
class DocumentPayload:

    content_type: ContentType = ContentType.NONE
    content: Any = None
    filename: str = None
    filename_values: Mapping[str, Any] = None
    chunk_id = None

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
    document_index_key: str = None
    document_index_sep: str = '\t'

    memory_store: Mapping[str, Any] = field(default_factory=dict)
    pos_schema_name: str = field(default="Universal")

    filenames: List[str] = None
    metadata: List[Dict[str, Any]] = None

    # FIXME: Move to document_index_proxy object
    primary_document_index: pd.DataFrame = None  # Given index i.e. DataFrane or loaded given  filenames
    secondary_document_index: pd.DataFrame = None  # Index reconstructed from source (filename, filename fields)
    consolidated_document_index: pd.DataFrame = None  # Merged index (if both exists)

    _document_index_lookup: Mapping[str, Dict[str, Any]] = None

    # NOT USED: token2id: Mapping = None
    # NOT USED: extract_tokens_opts: ExtractTaggedTokensOpts = None
    # NOT USED: tokens_transform_opts: TokensTransformOpts = None
    # NOT USED: extract_opts: Mapping = None

    def get(self, key: str, default=None):
        return self.memory_store.get(key, default)

    def put(self, key: str, value: Any):
        self.memory_store[key] = value

    @property
    def document_index(self) -> pd.DataFrame:

        if self.primary_document_index is None:
            if self.document_index_source is not None:
                self.primary_document_index = self.load_primary_index()

        if self.consolidated_document_index is None:
            if self.primary_document_index is not None and self.secondary_document_index is not None:
                self.consolidated_document_index = consolidate_document_index(
                    index=self.primary_document_index,
                    reader_index=self.secondary_document_index,
                )

        if self.consolidated_document_index is not None:
            return self.consolidated_document_index

        if self.primary_document_index is not None:
            return self.primary_document_index

        return self.secondary_document_index

    def load_primary_index(self) -> pd.DataFrame:

        if self.document_index_source is None:
            return None

        if isinstance(self.document_index_source, pd.DataFrame):
            return self.document_index_source

        if isinstance(self.document_index_source, str):
            return load_document_index(
                self.document_index_source,
                key_column=self.document_index_key,
                sep=self.document_index_sep,
            )

        return None

    def document_lookup(self, filename: str) -> Dict[str, Any]:
        if self._document_index_lookup is None:
            self._document_index_lookup = {x['filename']: x for x in self.document_index.to_dict(orient='record')}
        return self._document_index_lookup.get(filename, None)


@dataclass
class ITask(abc.ABC):

    pipeline: pipelines.CorpusPipeline = None
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

    def hookup(self, pipeline: pipelines.AnyPipeline) -> ITask:
        self.pipeline = pipeline
        return self

    @property
    def document_index(self) -> pd.DataFrame:
        return self.pipeline.payload.document_index

    def input_type_guard(self, content_type):
        if self.in_content_type is None or self.in_content_type == ContentType.NONE:
            return
        if isinstance(self.in_content_type, ContentType):
            if self.in_content_type == ContentType.ANY:
                return
            if self.in_content_type == content_type:
                return
        if isinstance(
            self.in_content_type,
            (list, tuple),
        ):
            if content_type in self.in_content_type:
                return
        raise PipelineError("content type not valid for task")
