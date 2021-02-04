from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Sequence, Union

import pandas as pd
from penelope.corpus import consolidate_document_index, load_document_index, update_document_index_properties
from penelope.corpus.readers import TextSource
from penelope.utility import Known_PoS_Tag_Schemes, PoS_Tag_Scheme, strip_path_and_extension

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
    CO_OCCURRENCE_DATAFRAME = 14
    STREAM = 15


@dataclass
class DocumentPayload:

    content_type: ContentType = ContentType.NONE
    filename: str = None
    content: Any = None
    filename_values: Mapping[str, Any] = None
    chunk_id = None
    previous_content_type: ContentType = field(default=ContentType.NONE, init=False)
    property_bag: dict = field(default_factory=dict, init=False)

    def update(self, content_type: ContentType, content: Any) -> "DocumentPayload":
        self.previous_content_type = self.content_type
        self.content_type = content_type
        self.content = content
        return self

    def update_properties(self, **properties) -> "DocumentPayload":
        """Save document properties to property bag"""
        self.property_bag.update(properties)
        return self

    @property
    def document_name(self):
        return strip_path_and_extension(self.filename)

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

    source_folder: str = None
    source: TextSource = None
    document_index_source: Union[str, pd.DataFrame] = None
    document_index_sep: str = '\t'

    memory_store: Mapping[str, Any] = field(default_factory=dict)
    pos_schema_name: str = field(default="Universal")
    _pos_schema: str = field(default=None, init=False)

    filenames: List[str] = None
    metadata: List[Dict[str, Any]] = None
    token2id: Mapping[str, int] = None
    effective_document_index: pd.DataFrame = None

    # FIXME: Move to document_index_proxy object?

    _document_index_lookup: Mapping[str, Dict[str, Any]] = None

    @property
    def document_index(self) -> pd.DataFrame:
        if self.effective_document_index is None:
            if isinstance(self.document_index_source, pd.DataFrame):
                self.effective_document_index = self.document_index_source
            elif isinstance(self.document_index_source, str):
                self.effective_document_index = load_document_index(
                    filename=self.document_index_source,
                    sep=self.document_index_sep,
                )
        return self.effective_document_index

    @property
    def props(self) -> Dict[str, Any]:
        return dict(
            source=self.source if isinstance(self.source, str) else 'object',
            document_index_source=self.document_index_source
            if isinstance(self.document_index_source, str)
            else 'object',
            pos_schema_name=self.pos_schema_name,
        )

    def get(self, key: str, default=None):
        return self.memory_store.get(key, default)

    def put(self, key: str, value: Any) -> "PipelinePayload":
        self.memory_store[key] = value
        return self

    def put2(self, **kwargs) -> "PipelinePayload":
        for key, value in kwargs.items():
            self.memory_store[key] = value
        return self

    def set_reader_index(self, reader_index: pd.DataFrame) -> "PipelinePayload":
        if self.document_index is None:
            self.effective_document_index = reader_index
        else:
            self.effective_document_index = consolidate_document_index(
                document_index=self.document_index,
                reader_index=reader_index,
            )
        return self

    def document_lookup(self, document_name: str) -> Dict[str, Any]:
        return self.document_index.loc[strip_path_and_extension(document_name)]

    @property
    def pos_schema(self) -> PoS_Tag_Scheme:

        if self._pos_schema is None:
            self._pos_schema = Known_PoS_Tag_Schemes.get(self.pos_schema_name, None)
            if self._pos_schema is None:
                raise PipelineError("expected PoS schema found None")

        return self._pos_schema

    def update_document_properties(self, document_name: str, **properties):
        """Updates document index with given property values"""
        update_document_index_properties(
            self.document_index,
            document_name=document_name,
            property_bag=properties,
        )

    @property
    def tagged_columns_names(self) -> dict:
        return {k: v for k, v in self.memory_store.items() if k in ['text_column', 'pos_column', 'lemma_column']}

    def extend(self, _: DocumentPayload):
        """Add properties of `other` to self. Used when combining two pipelines"""
        ...


@dataclass
class ITask(abc.ABC):

    pipeline: pipelines.CorpusPipeline = None
    instream: Iterable[DocumentPayload] = None

    in_content_type: Union[ContentType, Sequence[ContentType]] = field(init=False, default=None)
    out_content_type: ContentType = field(init=False, default=None)

    def chain(self) -> ITask:
        prior_task: ITask = self.pipeline.get_prior_to(self)
        if prior_task is not None:
            self.instream = prior_task.outstream()
        return self

    def setup(self) -> ITask:
        return self

    @abc.abstractmethod
    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return payload

    def process(self, payload: DocumentPayload) -> DocumentPayload:
        self.input_type_guard(payload.content_type)
        return self.process_payload(payload)

    def enter(self):
        return

    def exit(self):
        return

    def outstream(self) -> Iterable[DocumentPayload]:
        if self.instream is None:
            raise PipelineError("No instream specified. Have you loaded a corpus source?")

        self.enter()
        for payload in self.instream:
            yield self.process(payload)
        self.exit()

    def hookup(self, pipeline: pipelines.AnyPipeline) -> ITask:
        self.pipeline = pipeline
        return self

    @property
    def document_index(self) -> pd.DataFrame:
        return self.pipeline.payload.document_index

    def input_type_guard(self, content_type) -> None:
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

    def update_document_properties(self, payload: DocumentPayload, **properties) -> None:
        """Stores document properties to document index"""
        payload.update_properties(**properties)
        self.pipeline.payload.update_document_properties(payload.document_name, **properties)


DocumentTagger = Callable[[DocumentPayload, List[str], Dict[str, Any]], pd.DataFrame]
