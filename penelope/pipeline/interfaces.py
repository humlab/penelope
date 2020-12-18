from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence, Union

import pandas as pd
from penelope.corpus import consolidate_document_index, load_document_index
from penelope.corpus.readers import TextSource
from penelope.utility import strip_path_and_extension
from penelope.utility.pos_tags import Known_PoS_Tag_Schemes, PoS_Tag_Scheme

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
    statistics: Mapping[str, int] = field(default=None, init=False)

    def update(self, content_type: ContentType, content: Any):
        self.previous_content_type = self.content_type
        self.content_type = content_type
        self.content = content
        return self

    def update_statistics(self, pos_statistics: Mapping[str, int], n_tokens: int):

        self.statistics = {
            'document_name': self.document_name,
            **pos_statistics.to_dict(),
            **dict(
                n_raw_tokens=pos_statistics[~(pos_statistics.index == 'Delimiter')].sum(),
                n_tokens=n_tokens,
            ),
        }

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
    document_index_key: str = None
    document_index_sep: str = '\t'

    memory_store: Mapping[str, Any] = field(default_factory=dict)
    pos_schema_name: str = field(default="Universal")
    _pos_schema: str = field(default=None, init=False)

    filenames: List[str] = None
    metadata: List[Dict[str, Any]] = None
    token2id: Mapping[str, int] = None

    _document_index: pd.DataFrame = None
    # FIXME: Move to document_index_proxy object?

    _document_index_lookup: Mapping[str, Dict[str, Any]] = None

    @property
    def document_index(self) -> pd.DataFrame:
        if self._document_index is None:
            if isinstance(self.document_index_source, pd.DataFrame):
                self._document_index = self.document_index_source
            elif isinstance(self.document_index_source, str):
                self._document_index = load_document_index(
                    filename=self.document_index_source,
                    key_column=self.document_index_key,
                    sep=self.document_index_sep,
                )
        return self._document_index

    @property
    def props(self) -> Dict[str, Any]:
        return dict(
            source=self.source if isinstance(self.source, str) else 'object',
            document_index_source=self.document_index_source
            if isinstance(self.document_index_source, str)
            else 'object',
            document_index_key=self.document_index_key,
            pos_schema_name=self.pos_schema_name,
        )

    def get(self, key: str, default=None):
        return self.memory_store.get(key, default)

    def put(self, key: str, value: Any) -> "PipelinePayload":
        self.memory_store[key] = value
        return self

    def set_reader_index(self, reader_index: pd.DataFrame) -> "PipelinePayload":
        if self._document_index is None:
            self._document_index = reader_index
        else:
            self._document_index = consolidate_document_index(
                document_index=self._document_index,
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
