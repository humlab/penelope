import contextlib
import glob
import os
import shutil
import zipfile
from contextlib import suppress
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Container, Dict, Iterable, List, Optional, Union

import pandas as pd
from loguru import logger
from penelope import utility
from penelope.corpus import (
    DocumentIndex,
    ITokenizedCorpus,
    Token2Id,
    TokensTransformer,
    VectorizedCorpus,
    VectorizeOpts,
    default_tokenizer,
)
from penelope.corpus.readers import (
    GLOBAL_TF_THRESHOLD_MASK_TOKEN,
    ExtractTaggedTokensOpts,
    TextReader,
    TextReaderOpts,
    TextSource,
    TextTransformer,
    TextTransformOpts,
)
from penelope.corpus.readers.tng import CorpusReader, create_sparv_xml_corpus_reader
from penelope.utility import PropertyValueMaskingOpts, replace_extension, strip_paths
from tqdm.auto import tqdm

from . import checkpoint as cp
from . import convert
from .interfaces import ContentType, DocumentPayload, DocumentTagger, ITask, PipelineError
from .tagged_frame import TaggedFrame
from .tasks_mixin import CountTaggedTokensMixIn, DefaultResolveMixIn, TransformTokensMixIn, VocabularyIngestMixIn


@dataclass
class LoadText(DefaultResolveMixIn, ITask):
    """Loads a text source into the pipeline.
    Note that this task can handle more kinds of source than "Checkpoint"
    Also loads a document_index, and/or extracts value fields from filenames
    """

    source: TextSource = None
    reader_opts: TextReaderOpts = None
    transform_opts: TextTransformOpts = None
    text_reader: TextReader = field(init=False, default=None)

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TEXT

    def setup(self) -> ITask:
        super().setup()

        self.transform_opts = self.transform_opts or TextTransformOpts()

        if self.source is not None:
            self.pipeline.payload.source = self.source

        self.text_reader: TextReader = TextReader.create(
            source=self.pipeline.payload.source,
            reader_opts=self.reader_opts,
            transform_opts=self.transform_opts,
        )

        self.pipeline.payload.set_reader_index(self.text_reader.document_index)
        self.pipeline.payload.metadata = self.text_reader.metadata

        self.pipeline.put("text_reader_opts", self.reader_opts.props)
        self.pipeline.put("text_transform_opts", self.transform_opts.props)

        return self

    def create_instream(self) -> Iterable[DocumentPayload]:
        return (
            DocumentPayload(filename=filename, content_type=ContentType.TEXT, content=text)
            for filename, text in self.text_reader
        )

    def process_stream(self) -> Iterable[DocumentPayload]:
        self.instream = self.create_instream()
        return super().process_stream()


@dataclass
class Tqdm(ITask):

    tbar = None
    desc: str = None

    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.ANY

    def setup(self) -> ITask:
        super().setup()
        self.tbar = tqdm(
            desc=self.desc,
            position=0,
            leave=True,
            total=len(self.document_index) if self.document_index is not None else None,
        )
        return self

    def process_payload(self, payload: DocumentPayload) -> Any:
        self.tbar.update()
        return payload


@dataclass
class Passthrough(DefaultResolveMixIn, ITask):
    pass


@dataclass
class Project(ITask):

    project: Callable[[DocumentPayload], Any] = None

    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.ANY

    def process_payload(self, payload: DocumentPayload) -> Any:
        return self.project(payload)


@dataclass
class ToContent(ITask):
    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = Any

    def process_payload(self, payload: DocumentPayload) -> Any:
        return payload.content


@dataclass
class ToDocumentContentTuple(ITask):
    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.DOCUMENT_CONTENT_TUPLE

    def process_payload(self, payload: DocumentPayload) -> Any:
        return payload.update(
            self.out_content_type,
            content=(
                payload.filename,
                payload.content,
            ),
        )


@dataclass
class Checkpoint(DefaultResolveMixIn, ITask):

    filename: str = None
    checkpoint_opts: cp.CheckpointOpts = None

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("checkpoint_file", self.filename)
        self.checkpoint_opts = self.checkpoint_opts or self.pipeline.config.checkpoint_opts
        self.pipeline.put("checkpoint_opts", self.checkpoint_opts)
        return self

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS, ContentType.TAGGED_FRAME]
        self.out_content_type = ContentType.PASSTHROUGH

    def create_instream(self) -> Iterable[DocumentPayload]:
        return self._load_payload_stream() if os.path.isfile(self.filename) else self._store_payload_stream()

    def process_stream(self) -> Iterable[DocumentPayload]:
        self.instream = self.create_instream()
        return super().process_stream()

    def _load_payload_stream(self):
        checkpoint_data: cp.CheckpointData = cp.load_checkpoint(
            self.filename,
            checkpoint_opts=self.checkpoint_opts,
        )
        self.pipeline.payload.effective_document_index = checkpoint_data.document_index
        self.out_content_type = checkpoint_data.content_type

        payload_stream = checkpoint_data.create_stream()
        return payload_stream

    def _store_payload_stream(self):
        self.out_content_type = self.get_out_content_type()
        checkpoint_opts = self.checkpoint_opts.as_type(self.out_content_type)
        payload_stream = cp.store_checkpoint(
            checkpoint_opts=checkpoint_opts,
            target_filename=self.filename,
            document_index=self.document_index,
            payload_stream=self.instream,
        )
        return payload_stream

    def get_out_content_type(self):
        prior_content_type = self.pipeline.get_prior_content_type(self)
        if prior_content_type == ContentType.NONE:
            raise PipelineError(
                "Checkpoint file removed OR pipeline setup error. Checkpoint file does not exist AND checkpoint task has no prior task"
            )
        return prior_content_type


@dataclass
class SaveTaggedCSV(Checkpoint):
    """Stores sequence of tagged data frame documents to archive. """

    filename: str = None
    checkpoint_opts: cp.CheckpointOpts = None

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGED_FRAME
        self.out_content_type = ContentType.TAGGED_FRAME

    def process_stream(self) -> Iterable[DocumentPayload]:
        for payload in self._store_payload_stream():
            yield payload

    def get_out_content_type(self):
        return self.out_content_type


@dataclass
class LoadTaggedCSV(CountTaggedTokensMixIn, DefaultResolveMixIn, ITask):
    """Loads CSV files stored in a ZIP as Pandas data frames. """

    filename: str = None
    checkpoint_opts: Optional[cp.CheckpointOpts] = None
    extra_reader_opts: Optional[TextReaderOpts] = None  # Use if e.g. document index should be created
    checkpoint_data: cp.CheckpointData = field(default=None, init=None, repr=None)

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TAGGED_FRAME

    def setup(self) -> ITask:
        super().setup()

        self.checkpoint_opts = self.checkpoint_opts or self.pipeline.config.checkpoint_opts
        self.checkpoint_data: cp.CheckpointData = self.load_checkpoint()
        self.pipeline.payload.set_reader_index(self.checkpoint_data.document_index)

        self.pipeline.put("reader_opts", self.extra_reader_opts.props)
        self.pipeline.put("checkpoint_opts", self.checkpoint_opts.props)

        return self

    def create_instream(self) -> Iterable[DocumentPayload]:
        return self.checkpoint_data.create_stream()

    def process_stream(self) -> Iterable[DocumentPayload]:
        self.instream = self.create_instream()
        return super().process_stream()

    def load_checkpoint(self) -> cp.CheckpointData:
        checkpoint_data: cp.CheckpointData = cp.load_checkpoint(
            self.filename,
            checkpoint_opts=self.checkpoint_opts,
            reader_opts=self.extra_reader_opts,
        )
        return checkpoint_data

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        self.register_token_counts(payload)
        return payload


FEATHER_DOCUMENT_INDEX_NAME = 'document_index.feathering'


@dataclass
class CheckpointFeather(DefaultResolveMixIn, ITask):
    """Creates a feather checkpoint. """

    folder: str = None
    force: bool = field(default=False)

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGED_FRAME
        self.out_content_type = ContentType.TAGGED_FRAME

    def enter(self):
        if self.force:
            with contextlib.suppress(Exception):
                shutil.rmtree(self.folder, ignore_errors=True)

    def process_stream(self) -> Iterable[DocumentPayload]:
        task_cls = ReadFeather if os.path.isdir(self.folder) else WriteFeather
        task: ITask = task_cls(folder=self.folder, pipeline=self.pipeline, instream=self.instream)
        return task.outstream()

    @staticmethod
    def read_document_index(folder: str) -> DocumentIndex:
        filename = os.path.join(folder, FEATHER_DOCUMENT_INDEX_NAME)
        if os.path.isfile(filename):
            document_index: DocumentIndex = pd.read_feather(filename).set_index('document_name', drop=False)
            if '' in document_index.columns:
                document_index.drop(columns='', inplace=True)
            return document_index
        return None

    @staticmethod
    def write_document_index(folder: str, document_index: DocumentIndex):
        filename = os.path.join(folder, FEATHER_DOCUMENT_INDEX_NAME)
        if document_index is not None:
            document_index.reset_index().to_feather(filename, compression="lz4")


@dataclass
class WriteFeather(ITask):
    """Stores sequence of tagged data frame documents to archive. """

    folder: str = None

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGED_FRAME
        self.out_content_type = ContentType.TAGGED_FRAME

    def enter(self):
        os.makedirs(self.folder, exist_ok=True)

    def process_payload(self, payload: DocumentPayload) -> Iterable[DocumentPayload]:
        tagged_frame: TaggedFrame = payload.content
        filename = os.path.join(self.folder, replace_extension(payload.filename, ".feather"))
        tagged_frame.to_feather(filename, compression="lz4")
        return payload

    def exit(self):
        CheckpointFeather.write_document_index(self.folder, self.document_index)


@dataclass
class ReadFeather(DefaultResolveMixIn, ITask):
    """Stores sequence of tagged data frame documents to archive. """

    folder: str = None

    document_index_filename: str = field(init=False, default=None)

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TAGGED_FRAME
        self.document_index_filename = os.path.join(self.folder, FEATHER_DOCUMENT_INDEX_NAME)

    def create_instream(self) -> Iterable[DocumentPayload]:
        pattern: str = os.path.join(self.folder, "*.feather")
        for path in sorted(glob.glob(pattern)):
            tagged_frame = pd.read_feather(path)
            filename = strip_paths(path)
            yield DocumentPayload(
                content_type=ContentType.TAGGED_FRAME,
                content=tagged_frame,
                filename=replace_extension(filename, ".csv"),
            )

    def process_stream(self) -> Iterable[DocumentPayload]:
        self.instream = self.create_instream()
        return super().process_stream()

    def enter(self):
        document_index = CheckpointFeather.read_document_index(self.folder)
        if document_index is not None:
            self.pipeline.payload.effective_document_index = document_index


@dataclass
class LoadTaggedXML(CountTaggedTokensMixIn, ITask):
    """Loads Sparv export documents stored as individual XML files in a ZIP-archive into a Pandas data frames. """

    filename: str = None
    reader_opts: TextReaderOpts = None
    corpus_reader: CorpusReader = field(default=None, init=None, repr=None)

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TAGGED_FRAME

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("reader_opts", self.reader_opts.props)
        self.corpus_reader = create_sparv_xml_corpus_reader(
            source_path=self.filename or self.pipeline.payload.source,
            reader_opts=self.reader_opts or self.pipeline.config.text_reader_opts,
            sparv_version=int(self.pipeline.payload.get("sparv_version", 0)),
            content_type="pandas",
        )
        self.pipeline.payload.set_reader_index(self.corpus_reader.document_index)

    def create_instream(self) -> Iterable[DocumentPayload]:
        for document, content in self.corpus_reader:
            yield DocumentPayload(
                content_type=ContentType.TAGGED_FRAME,
                filename=document,
                content=content,
                filename_values=None,
            )

    def process_stream(self) -> Iterable[DocumentPayload]:
        self.instream = self.create_instream()
        return super().process_stream()

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        self.register_token_counts(payload)
        return payload


@dataclass
class TextToTokens(TransformTokensMixIn, ITask):
    """Extracts tokens from payload.content, optinally transforming"""

    tokenize: Callable[[str], List[str]] = None
    text_transform_opts: TextTransformOpts = None
    _text_transformer: TextTransformer = field(init=False)

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("text_transform_opts", self.text_transform_opts)
        return self

    def __post_init__(self):

        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS]
        self.out_content_type = ContentType.TOKENS
        self.tokenize = self.tokenize or default_tokenizer

        if self.text_transform_opts is not None:
            self._text_transformer = TextTransformer(text_transform_opts=self.text_transform_opts)

        self.setup_transform()

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        if self.in_content_type == ContentType.TOKENS:
            tokens = payload.content
        else:
            if self._text_transformer is not None:
                self.tokenize(self._text_transformer.transform(payload.content))
            tokens = self.tokenize(payload.content)

        tokens = self.transform(tokens)

        return payload.update(self.out_content_type, tokens)


@dataclass
class ToTaggedFrame(CountTaggedTokensMixIn, ITask):

    attributes: List[str] = None
    attribute_value_filters: Dict[str, Any] = None
    tagger: DocumentTagger = None

    def setup(self) -> ITask:
        self.pipeline.put("tagged_attributes", self.attributes)
        return self

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS]
        self.out_content_type = ContentType.TAGGED_FRAME

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

        tagged_frame: TaggedFrame = self.tagger(
            payload=payload,
            attributes=self.attributes,
            attribute_value_filters=self.attribute_value_filters,
        )

        payload = payload.update(self.out_content_type, tagged_frame)

        self.register_token_counts(payload)

        return payload


@dataclass
class TaggedFrameToTokens(
    CountTaggedTokensMixIn,
    VocabularyIngestMixIn,
    TransformTokensMixIn,
    ITask,
):
    """Extracts text from payload.content based on annotations etc. """

    extract_opts: ExtractTaggedTokensOpts = None
    filter_opts: PropertyValueMaskingOpts = None

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGED_FRAME
        self.out_content_type = ContentType.TOKENS

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("extract_opts", self.extract_opts)
        self.pipeline.put("filter_opts", self.filter_opts)
        self.pipeline.put("transform_opts", self.transform_opts)

        return self

    def enter(self) -> ITask:  # pylint: disable=useless-super-delegation
        super().enter()

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

        if self.pipeline.get('pos_column', None) is None:
            raise PipelineError("expected `pos_column` in `payload.memory_store` found None")

        tokens: Iterable[str] = convert.tagged_frame_to_tokens(
            doc=payload.content,
            extract_opts=self.extract_opts,
            filter_opts=self.filter_opts,
            **(self.pipeline.payload.tagged_columns_names or {}),
            transform_opts=self.transform_opts,
        )

        tokens = list(tokens)

        self.ingest(tokens)

        self.update_document_properties(payload, n_tokens=len(tokens))  # , n_raw_tokens=len(payload.content))

        return payload.update(self.out_content_type, tokens)


@dataclass
class TapStream(CountTaggedTokensMixIn, ITask):
    """Taps content into zink. """

    target: str = None
    tag: str = None
    enabled: bool = False
    zink: zipfile.ZipFile = None

    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.PASSTHROUGH

    def enter(self):
        logger.info(f"Tapping stream to {self.target}")
        self.zink = zipfile.ZipFile(self.target, "w")  # pylint: disable=consider-using-with

    def exit(self):
        self.zink.close()

    def store(self, payload: DocumentPayload) -> DocumentPayload:

        with suppress(Exception):
            content: str = somewhat_generic_serializer(payload.content)
            if content is not None:
                self.zink.writestr(f"{self.tag}__{payload.filename}", content)

        return payload

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return self.store(payload)


def somewhat_generic_serializer(content: Any) -> Optional[str]:

    if isinstance(content, pd.core.api.DataFrame):
        return content.to_csv(sep='\t')

    if isinstance(content, list):
        return ' '.join(content)

    if isinstance(content, str):
        return content

    if isinstance(content, tuple) and len(content) == 2:
        return somewhat_generic_serializer(content[1])

    return None


@dataclass
class TokensTransform(TransformTokensMixIn, ITask):
    """Transforms tokens payload.content"""

    def setup(self) -> ITask:
        super().setup()
        self.setup_transform()
        return self

    def __post_init__(self):
        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.TOKENS
        self.transformer = TokensTransformer(transform_opts=self.transform_opts)

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        tokens: List[str] = self.transform(payload.content)
        # FIXME: call self.update_document_properties(payload, n_tokens=len(tokens))??
        return payload.update(self.out_content_type, tokens)

    def add(self, transform: Callable[[List[str]], List[str]]) -> "TokensTransform":
        self.transformer.add(transform)
        return self


@dataclass
class TokensToText(ITask):
    """Extracts text from payload.content"""

    def __post_init__(self):
        self.in_content_type = [ContentType.TOKENS, ContentType.TEXT]
        self.out_content_type = ContentType.TEXT

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return payload.update(self.out_content_type, utility.to_text(payload.content))


@dataclass
class TextToDTM(ITask):
    def __post_init__(self):
        self.in_content_type = ContentType.DOCUMENT_CONTENT_TUPLE
        self.out_content_type = ContentType.VECTORIZED_CORPUS

    vectorize_opts: VectorizeOpts = None

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("vectorize_opts", self.vectorize_opts)
        return self

    def process_stream(self) -> VectorizedCorpus:
        corpus = convert.to_vectorized_corpus(
            stream=self.instream,
            vectorize_opts=self.vectorize_opts,
            document_index=lambda: self.pipeline.payload.document_index,
        )
        yield DocumentPayload(content_type=ContentType.VECTORIZED_CORPUS, content=corpus)

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None


# FIXME #115 Enable optional one-pass creation of vocabulary and TF frequencies
@dataclass
class Vocabulary(DefaultResolveMixIn, ITask):
    class TokenType(IntEnum):
        Text = 1
        Lemma = 2

    token2id: Token2Id = None
    token_type: Optional[TokenType] = None
    progress: bool = False
    close: bool = True
    tf_threshold: int = None
    tf_keeps: Container[Union[int, str]] = 1

    target: str = field(init=False, default="")

    def __post_init__(self):
        self.in_content_type = [ContentType.TOKENS, ContentType.TAGGED_FRAME]
        self.out_content_type = ContentType.PASSTHROUGH

    def setup(self) -> ITask:
        self.target = self.get_column_name(self.token_type)
        if self.in_content_type == ContentType.TAGGED_FRAME:
            if self.token_type is None:
                raise ValueError("token_type text or lemma not specfied")

        self.token2id: Token2Id = Token2Id()
        self.pipeline.payload.token2id = self.token2id
        return self

    def enter(self):

        instream = tqdm(self.instream, desc="Vocab:") if self.progress else self.instream

        ingest = self.token2id.ingest

        ingest(["*", GLOBAL_TF_THRESHOLD_MASK_TOKEN])

        for payload in instream:
            ingest(self.tokens_stream(payload))

        if self.tf_threshold and self.tf_threshold > 1:
            self.token2id.compress(tf_threshold=self.tf_threshold, inplace=True, keeps=self.tf_keeps)
        elif self.close:
            self.token2id.close()

        self.reset()
        return self

    def tokens_stream(self, payload: DocumentPayload) -> Iterable[str]:
        if payload.content_type == ContentType.TOKENS:
            return payload.content
        if self.token_type == Vocabulary.TokenType.Lemma:
            tokens = payload.content[self.target].str.lower()
        else:
            tokens = payload.content[self.target]
        return tokens

    def get_column_name(self, token_type: TokenType) -> str:
        if token_type == Vocabulary.TokenType.Lemma:
            return self.pipeline.payload.memory_store.get("lemma_column")
        return self.pipeline.payload.memory_store.get("text_column")

    def reset(self) -> None:
        self.chain()


@dataclass
class ChunkTokens(ITask):
    chunk_size: int = None

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("chunk_size", self.chunk_size)
        return self

    def __post_init__(self):
        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.TOKENS

    def process_stream(self) -> Iterable[DocumentPayload]:

        for payload in self.instream:
            tokens = payload.content
            if len(payload.content) < self.chunk_size:
                yield payload
            else:
                for chunk_id, i in enumerate(range(0, len(tokens), self.chunk_size)):
                    yield DocumentPayload(
                        filename=payload.filename,
                        content_type=ContentType.TOKENS,
                        content=tokens[i : i + self.chunk_size],
                        chunk_id=chunk_id,
                    )


@dataclass
class WildcardTask(ITask):
    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.NONE

    def abort(self):
        raise PipelineError("fatal: not instantiated wildcard task encountered. Please check configuration!")

    def process_stream(self) -> Iterable[DocumentPayload]:
        self.abort()

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        self.abort()


@dataclass
class LoadTokenizedCorpus(DefaultResolveMixIn, ITask):
    """Loads Sparv export documents stored as individual XML files in a ZIP-archive into a Pandas data frames. """

    corpus: ITokenizedCorpus = None

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TOKENS

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.payload.set_reader_index(self.corpus.document_index)

    def create_instream(self) -> Iterable[DocumentPayload]:
        return (
            DocumentPayload(
                content_type=self.out_content_type,
                filename=filename,
                content=content,
                filename_values=None,
            )
            for filename, content in self.corpus
        )

    def process_stream(self) -> Iterable[DocumentPayload]:
        self.instream = self.create_instream()
        return super().process_stream()

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        self.update_document_properties(payload, n_tokens=len(payload.content))
        return payload


# @dataclass
# class FilterTaggedFrame(CountTaggedTokensMixIn, ITask):
#     """Filters tagged frame text from payload.content based on annotations etc. """

#     extract_opts: ExtractTaggedTokensOpts = None
#     filter_opts: PropertyValueMaskingOpts = None

#     def __post_init__(self):
#         self.in_content_type = ContentType.TAGGED_FRAME
#         self.out_content_type = ContentType.TOKENS

#     def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

#         if self.pipeline.get('pos_column', None) is None:
#             raise PipelineError("expected `pos_column` in `payload.memory_store` found None")

#         tokens: Iterable[str] = convert.tagged_frame_to_tokens(
#             doc=payload.content,
#             extract_opts=self.extract_opts,
#             filter_opts=self.filter_opts,
#             **(self.pipeline.payload.tagged_columns_names or {}),
#         )

#         tokens = list(tokens)

#         self.update_document_properties(payload, n_tokens=len(tokens))

#         return payload.update(self.out_content_type, tokens)


# class Split(ITask):

#     partioner: Callable = None

#     def __post_init__(self):
#         self.in_content_type = ContentType.ANY
#         self.out_content_type = ContentType.STREAM

#     def setup(self) -> ITask:
#         super().setup()

#     def process_payload(self, payload: DocumentPayload) -> Any:
#         raise NotImplementedError()

#     def process_stream(self) -> Iterable[DocumentPayload]:
#         raise NotImplementedError()

# class Reduce(ITask):

#     reducer: Callable = None
#     reducer: Callable = None

#     def __post_init__(self):
#         self.in_content_type = ContentType.ANY
#         self.out_content_type = ContentType.STREAM

#     def setup(self) -> ITask:
#         super().setup()

#     def process_payload(self, payload: DocumentPayload) -> Any:
#         raise NotImplementedError()

#     def process_stream(self) -> Iterable[DocumentPayload]:
#         raise NotImplementedError()
