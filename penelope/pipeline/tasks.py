import contextlib
import itertools
import os
import shutil
import zipfile
from contextlib import suppress
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Container, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd
from loguru import logger
from penelope import utility
from penelope.corpus import (
    ITokenizedCorpus,
    Token2Id,
    TokensTransformer,
    VectorizedCorpus,
    VectorizeOpts,
    default_tokenizer,
)
from penelope.corpus.readers import (
    ExtractTaggedTokensOpts,
    TextReader,
    TextReaderOpts,
    TextSource,
    TextTransformer,
    TextTransformOpts,
)
from penelope.corpus.readers.tng import CorpusReader, create_sparv_xml_corpus_reader
from penelope.utility import PropertyValueMaskingOpts
from tqdm.auto import tqdm

from . import checkpoint as cp
from . import convert
from .interfaces import ContentType, DocumentPayload, DocumentTagger, ITask, PipelineError
from .tasks_mixin import CountTaggedTokensMixIn, DefaultResolveMixIn, TransformTokensMixIn, VocabularyIngestMixIn


class EmptyCheckPointError(PipelineError):
    ...


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

    def get_filenames(self) -> List[str]:
        return self.document_index['filename'].tolist()


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
    """Checkpoints stream to a single files archive"""

    filename: str = None
    checkpoint_opts: cp.CheckpointOpts = None
    force_checkpoint: bool = False

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("tagged_frames_filename", self.filename)
        self.checkpoint_opts = self.checkpoint_opts or self.pipeline.config.checkpoint_opts
        self.pipeline.put("checkpoint_opts", self.checkpoint_opts)
        return self

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS, ContentType.TAGGED_FRAME]
        self.out_content_type = ContentType.PASSTHROUGH

    def enter(self):
        if self.force_checkpoint:
            if os.path.isfile(self.filename):
                os.remove(self.filename)
            self.force_checkpoint = False

        return super().enter()

    def create_instream(self) -> Iterable[DocumentPayload]:
        return self._load_payload_stream() if os.path.isfile(self.filename) else self._store_payload_stream()

    def _load_payload_stream(self):
        checkpoint_data: cp.CheckpointData = cp.load_archive(self.filename, checkpoint_opts=self.checkpoint_opts)
        self.pipeline.payload.effective_document_index = checkpoint_data.document_index
        self.out_content_type = checkpoint_data.content_type
        return checkpoint_data.create_stream()

    def _store_payload_stream(self):
        self.out_content_type = self.get_out_content_type()
        checkpoint_opts = self.checkpoint_opts.as_type(self.out_content_type)
        return cp.store_archive(
            checkpoint_opts=checkpoint_opts,
            target_filename=self.filename,
            document_index=self.document_index,
            payload_stream=self.prior.outstream(),
        )

    def get_out_content_type(self):
        prior_content_type = self.pipeline.get_prior_content_type(self)
        if prior_content_type == ContentType.NONE:
            raise PipelineError(
                "Checkpoint file removed OR pipeline setup error. Checkpoint file does not exist AND checkpoint task has no prior task"
            )
        return prior_content_type


@dataclass
class SaveTaggedCSV(Checkpoint):
    """Stores sequence of tagged data frame to archive and optionally to FEATHER files. """

    filename: str = None
    checkpoint_opts: cp.CheckpointOpts = None

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGED_FRAME
        self.out_content_type = ContentType.TAGGED_FRAME

    def create_instream(self) -> Iterable[DocumentPayload]:
        return self._store_payload_stream()

    def get_out_content_type(self):
        return self.out_content_type


@dataclass
class LoadTaggedCSV(CountTaggedTokensMixIn, ITask):
    """Load Pandas data frames from folder (CSV, feather), ZIP archive. """

    filename: str = None
    checkpoint_opts: Optional[cp.CheckpointOpts] = None
    extra_reader_opts: Optional[TextReaderOpts] = None
    checkpoint_data: cp.CheckpointData = field(default=None, init=None, repr=None)
    stop_at_index: int = None

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TAGGED_FRAME

    def enter(self):

        if self.checkpoint_opts.feather_folder:
            os.makedirs(self.checkpoint_opts.feather_folder, exist_ok=True)

    def exit(self):

        if self.checkpoint_opts.feather_folder:
            # Check validity of document index, fillna()?
            cp.feather.write_document_index(self.checkpoint_opts.feather_folder, self.document_index)

    def setup(self) -> ITask:
        super().setup()

        self.checkpoint_opts = self.checkpoint_opts or self.pipeline.config.checkpoint_opts
        self.checkpoint_data: cp.CheckpointData = self.load_archive()

        document_index: pd.DataFrame = cp.feather.get_document_index(
            self.checkpoint_opts.feather_folder, self.checkpoint_data.document_index
        )

        self.pipeline.payload.set_reader_index(document_index)

        self.pipeline.put("reader_opts", self.extra_reader_opts.props)
        self.pipeline.put("checkpoint_opts", self.checkpoint_opts.props)

        return self

    def create_instream(self) -> Iterable[DocumentPayload]:

        if self.stop_at_index:
            logger.info(f"LoadTaggedCSV: will stop at index {self.stop_at_index}")
            return itertools.islice(self.checkpoint_data.create_stream(), 0, self.stop_at_index, 1)

        return self.checkpoint_data.create_stream()

    def load_archive(self) -> cp.CheckpointData:
        return cp.load_archive(
            self.filename,
            checkpoint_opts=self.checkpoint_opts,
            reader_opts=self.extra_reader_opts,
        )

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        self.register_token_counts(payload)
        return payload

    def get_filenames(self) -> List[str]:
        return self.checkpoint_data.filenames if bool(self.checkpoint_data) else []


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
                if os.path.isdir(self.folder):
                    shutil.rmtree(self.folder, ignore_errors=True)
        self.force = False

    def create_instream(self) -> Iterable[DocumentPayload]:
        return (
            ReadFeather(folder=self.folder, pipeline=self.pipeline)
            if cp.feather.document_index_exists(folder=self.folder)
            else WriteFeather(folder=self.folder, prior=self.prior, pipeline=self.pipeline, force=self.force)
        ).outstream()


@dataclass
class WriteFeather(ITask):
    """Stores sequence of tagged data frame documents to archive. """

    folder: str = None
    force: bool = False

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGED_FRAME
        self.out_content_type = ContentType.TAGGED_FRAME

    def enter(self):
        os.makedirs(self.folder, exist_ok=True)

    def process_payload(self, payload: DocumentPayload) -> Iterable[DocumentPayload]:
        if self.force or not cp.feather.payload_exists(folder=self.folder, payload=payload):
            cp.feather.write_payload(folder=self.folder, payload=payload)
        return payload

    def exit(self):
        cp.feather.write_document_index(self.folder, self.document_index)


@dataclass
class ReadFeather(DefaultResolveMixIn, ITask):
    """Stores sequence of tagged data frame documents to archive. """

    folder: str = None

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TAGGED_FRAME

    def enter(self):
        self.pipeline.payload.effective_document_index = cp.feather.read_document_index(self.folder)

    def process_stream(self) -> Iterable[DocumentPayload]:
        return (
            cp.feather.read_payload(os.path.join(self.folder, filename))
            for filename in self.document_index.filename.tolist()
        )


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
        return (
            DocumentPayload(
                content_type=ContentType.TAGGED_FRAME,
                filename=document,
                content=content,
                filename_values=None,
            )
            for document, content in self.corpus_reader
        )

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        self.register_token_counts(payload)
        return payload

    def get_filenames(self) -> List[str]:
        return self.document_index['filename'].tolist()


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

        payload.update(
            self.out_content_type,
            self.tagger(
                payload=payload,
                attributes=self.attributes,
                attribute_value_filters=self.attribute_value_filters,
            ),
        )

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

    update_counts_on_exit: bool = True
    token_counts: dict = field(init=False, default_factory=dict)

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
            transform_opts=self.transform_opts,
            token2id=self.pipeline.payload.token2id,
        )

        tokens = list(tokens)

        if self.ingest_tokens:
            self.ingest(tokens)

        # if self.update_counts_on_exit:
        #     self.token_counts[payload.document_name] = len(tokens)
        # else:
        self.update_document_properties(payload, n_tokens=len(tokens))  # , n_raw_tokens=len(payload.content))

        return payload.update(self.out_content_type, tokens)

    def exit(self) -> ITask:  # pylint: disable=useless-super-delegation
        super().exit()
        # if self.update_counts_on_exit:
        #     self.update_document_index_key_values('n_tokens', self.token_counts)


@dataclass
class TapStream(ITask):
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


class AssertPayloadError(PipelineError):
    ...


# @dataclass
# class AssertPayloadContent(ITask):
#     """Test utility task: asserts payload content equals expected values """

#     expected_values: Iterable[Any] = None
#     comparer: Callable[[Any, Any], bool] = None
#     accept_fewer_expected_values: Callable[[Any, Any], bool] = False

#     _expected_values_iter: Iterator[Any] = field(init=False, default=None)

#     def __post_init__(self):
#         self.in_content_type = ContentType.ANY
#         self.out_content_type = ContentType.PASSTHROUGH

#     def enter(self):
#         self._expected_values_iter: Iterator[Any] = iter(self.expected_values)

# def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

#         try:

#             expected_value: Any = next(self._expected_values_iter)

#             if self.comparer and expected_value:
#                 if not self.comparer(payload.content, expected_value):
#                     logger.error(f"AssertPayloadContent: failed for document {payload.filename}:")
#                     logger.error(f"   content:\n{payload.content}")
#                     logger.error(f"  expected:\n{expected_value}")
#                     raise AssertPayloadError()

#         except StopIteration as x:
#             if not self.accept_fewer_expected_values:
#                 raise AssertPayloadError("AssertPayloadContent: to few expected values") from x

#     return payload


class AssertOnExitError(PipelineError):
    ...


@dataclass
class AssertOnExit(DefaultResolveMixIn, ITask):
    """Test utility task: asserts payload content equals expected values """

    exit_test: Callable[[Any, Any], bool] = None
    exit_test_args: Sequence[Any] = field(default_factory=list)

    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.PASSTHROUGH

    # def create_instream(self) -> Iterable[DocumentPayload]:
    #     return self.prior.create_instream()


@dataclass
class AssertOnPayload(ITask):
    """Test utility task: asserts payload content equals expected values """

    payload_test: Callable[[Any, DocumentPayload, Any], bool] = None
    payload_test_args: Sequence[Any] = field(default_factory=list)

    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.PASSTHROUGH

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        if not self.payload_test(self.pipeline, payload, *self.payload_test_args):
            raise AssertPayloadError()
        return payload

    # def create_instream(self) -> Iterable[DocumentPayload]:
    #     return self.prior.create_instream()


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
        # FIXME: call self.update_document_properties(payload, n_tokens=len(tokens))??
        return payload.update(self.out_content_type, self.transform(payload.content))

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
        yield DocumentPayload(
            content_type=ContentType.VECTORIZED_CORPUS,
            content=convert.to_vectorized_corpus(
                stream=self.create_instream(),
                vectorize_opts=self.vectorize_opts,
                document_index=lambda: self.pipeline.payload.document_index,
            ),
        )

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None


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
    tf_keeps: Container[Union[int, str]] = field(default_factory=set)

    target: str = field(init=False, default="")

    def __post_init__(self):
        self.in_content_type = [ContentType.TOKENS, ContentType.TAGGED_FRAME]
        self.out_content_type = ContentType.PASSTHROUGH
        self.tf_keeps = set(self.tf_keeps or [])

    def setup(self) -> ITask:
        self.target = self.get_column_name(self.token_type)

        # if self.pipeline.get_next_to(self).in_content_type == ContentType.TAGGED_FRAME:
        #     if self.token_type is None:
        #         raise ValueError("token_type text or lemma not specfied")

        self.token2id: Token2Id = self.token2id or Token2Id()
        self.pipeline.payload.token2id = self.token2id
        return self

    def enter(self):

        self.token2id.ingest(self.token2id.magic_tokens)
        self.tf_keeps |= self.token2id.magic_tokens

        for payload in self.prior.outstream(total=len(self.document_index.index), desc="Vocab"):
            self.token2id.ingest_stream([self.tokens_stream(payload)])

        if self.tf_threshold and self.tf_threshold > 1:
            """We don't need translation since vocab hasn't been used yet"""
            _ = self.token2id.compress(tf_threshold=self.tf_threshold, inplace=True, keeps=self.tf_keeps)

        if self.close:
            self.token2id.close()

        return self

    def tokens_stream(self, payload: DocumentPayload) -> Iterable[str]:

        if payload.content_type == ContentType.TOKENS:
            return payload.content

        if payload.recall('term_frequency'):
            return payload.recall('term_frequency')

        return (
            (x.lower() for x in payload.content[self.target])
            if self.token_type == Vocabulary.TokenType.Lemma
            else payload.content[self.target]
        )

    def get_column_name(self, token_type: TokenType) -> str:
        if token_type == Vocabulary.TokenType.Lemma:
            return self.pipeline.payload.memory_store.get("lemma_column")
        return self.pipeline.payload.memory_store.get("text_column")


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

        for payload in self.create_instream():
            tokens = payload.content
            if len(payload.content) < self.chunk_size:
                yield payload
            else:
                for chunk_id, i in enumerate(range(0, len(tokens), self.chunk_size)):
                    yield DocumentPayload(  # pylint: disable=unexpected-keyword-arg
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


class Split(ITask):
    ...


class Reduce(ITask):
    ...
