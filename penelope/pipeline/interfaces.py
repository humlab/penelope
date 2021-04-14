from __future__ import annotations

import abc
import os
from collections import defaultdict
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from penelope.corpus import (
    DocumentIndex,
    DocumentIndexHelper,
    consolidate_document_index,
    load_document_index,
    update_document_index_properties,
)
from penelope.corpus.readers import TextSource
from penelope.utility import Known_PoS_Tag_Schemes, PoS_Tag_Scheme, strip_path_and_extension
from penelope.utility.filename_utils import replace_path

from .tagged_frame import TaggedFrame

if TYPE_CHECKING:
    from . import pipelines


@unique
class ContentType(IntEnum):
    NONE = 0
    TAGGED_FRAME = 1
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
    TAGGED_ID_FRAME = 16


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

    # source_folder: str = None
    source: TextSource = None
    document_index_source: Union[str, DocumentIndex] = None
    document_index_sep: str = '\t'

    memory_store: Mapping[str, Any] = field(default_factory=dict)
    pos_schema_name: str = field(default="Universal")
    _pos_schema: str = field(default=None, init=False)

    filenames: List[str] = None
    metadata: List[Dict[str, Any]] = None
    token2id: Mapping[str, int] = None
    effective_document_index: DocumentIndex = None

    # FIXME: Move to document_index_proxy object?

    _document_index_lookup: Mapping[str, Dict[str, Any]] = None

    @property
    def document_index(self) -> DocumentIndex:
        if self.effective_document_index is None:
            if isinstance(self.document_index_source, DocumentIndex):
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

    def set_reader_index(self, reader_index: DocumentIndex) -> "PipelinePayload":
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

    def extend_document_index(self, other_index: DocumentIndex) -> "PipelinePayload":
        if self.effective_document_index is None:
            self.effective_document_index = other_index
        else:
            self.effective_document_index = (
                DocumentIndexHelper(self.effective_document_index).extend(other_index).document_index
            )
        return self

    @staticmethod
    def update_path(new_path: str, old_path: str, method: str) -> str:
        """Updates folder path or old_path, either by replacing existing path or by joining"""
        if method not in {"join", "replace"}:
            raise ValueError("only strategies `merge` or `replace` are allowed")
        if method == "join":
            return os.path.join(new_path, old_path)
        return replace_path(old_path, new_path)

    def folders(self, path: str, method: str = "replace") -> "PipelinePayload":
        """Replaces (any) existing source path specification for corpus/index to `path`"""

        if isinstance(self.document_index_source, str):
            self.document_index_source = self.update_path(path, self.document_index_source, method)
        if isinstance(self.source, str):
            self.source = self.update_path(path, self.source, method)

        return self

    def files(self, source: TextSource, document_index_source: Union[str, DocumentIndex]) -> "PipelinePayload":
        """Sets corpus source to `source` and document index source to `document_index_source`"""
        self.source = source
        self.document_index_source = document_index_source
        return self


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
        """Called prior to stream generation."""
        return

    def exit(self):
        """Called after stream has been generated."""
        return

    def process_stream(self) -> Iterable[DocumentPayload]:
        """Generates stream of payloads. Overridable. """

        if self.instream is None:
            raise PipelineError("No instream specified. Have you loaded a corpus source?")

        for payload in self.instream:
            yield self.process(payload)

    # FIXME #50 Make outstream non-overridable
    def outstream(self) -> Iterable[DocumentPayload]:
        """Returns stream of payloads. Non-overridable! """

        self.enter()
        for payload in self.process_stream():
            yield payload
        self.exit()

    def hookup(self, pipeline: pipelines.AnyPipeline) -> ITask:
        self.pipeline = pipeline
        return self

    @property
    def document_index(self) -> DocumentIndex:
        return self.pipeline.payload.document_index

    def input_type_guard(self, content_type) -> None:
        if self.in_content_type is None or self.in_content_type == ContentType.NONE:
            return
        if isinstance(self.in_content_type, ContentType):
            if self.in_content_type == ContentType.ANY:
                return
            if self.in_content_type == content_type:
                return
        if isinstance(self.in_content_type, (list, tuple)):
            if content_type in self.in_content_type:
                return
        raise PipelineError("content type not valid for task")

    def update_document_properties(self, payload: DocumentPayload, **properties) -> None:
        """Stores document properties to document index"""
        payload.update_properties(**properties)
        self.pipeline.payload.update_document_properties(payload.document_name, **properties)


DocumentTagger = Callable[[DocumentPayload, List[str], Dict[str, Any]], TaggedFrame]

# FIXME #46 Token2Id raises KeyError
class Token2Id(MutableMapping):
    """A token-to-id mapping (dictionary)"""

    def __init__(self, store: Optional[Union[dict, defaultdict]] = None):
        if isinstance(store, defaultdict):
            self.store = store
        elif isinstance(store, dict):
            self.store = defaultdict(int, self.store)
        else:
            self.store = store or defaultdict()
        self.store.default_factory = self.store.__len__

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key

    def ingest(self, tokens: Iterator[str]) -> "Token2Id":
        for token in tokens:
            _ = self.store[token]
        return self

    def close(self) -> "Token2Id":
        self.store.default_factory = None

    def open(self) -> "Token2Id":
        self.store.default_factory = self.__len__
        return self

    def id2token(self) -> dict:
        return {v: k for k, v in self.store.items()}
