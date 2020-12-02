import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional

from penelope.corpus import TokensTransformer, TokensTransformOpts, VectorizedCorpus, VectorizeOpts, default_tokenizer
from penelope.corpus.readers import TextReader, TextReaderOpts, TextSource, TextTransformer, TextTransformOpts
from penelope.utility import to_text
from tqdm.std import tqdm

from . import checkpoint, convert, interfaces
from .interfaces import ContentType


class DefaultResolveMixIn:
    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload


@dataclass
class LoadText(DefaultResolveMixIn, interfaces.ITask):
    """Loads a text source into the pipeline.
    Note that this task can handle more kinds of source than "Checkpoint"
    Also loads a document_index, and/or extracts value fields from filenames
    """

    source: TextSource = None
    reader_opts: TextReaderOpts = None
    transform_opts: TextTransformOpts = None

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TEXT

    def setup(self):
        super().setup()
        self.transform_opts = self.transform_opts or TextTransformOpts()
        if self.source is not None:
            self.pipeline.payload.source = self.source
        text_reader: TextReader = (
            self.pipeline.payload.source
            if isinstance(self.pipeline.payload.source, TextReader)
            else TextReader.create(
                source=self.pipeline.payload.source,
                reader_opts=self.reader_opts,
                transform_opts=self.transform_opts,
            )
        )
        self.pipeline.payload.secondary_document_index = text_reader.document_index
        self.pipeline.payload.metadata = text_reader.metadata
        self.pipeline.put("text_reader_opts", self.reader_opts.props)
        self.pipeline.put("text_transform_opts", self.transform_opts.props)

        self.instream = (
            interfaces.DocumentPayload(filename=filename, content_type=interfaces.ContentType.TEXT, content=text)
            for filename, text in text_reader
        )
        return self


@dataclass
class Tqdm(interfaces.ITask):

    tbar = None

    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.ANY

    def setup(self):
        super().setup()
        self.tbar = tqdm(
            position=0,
            leave=True,
            total=len(self.document_index) if self.document_index is not None else None,
        )
        return self

    def process_payload(self, payload: interfaces.DocumentPayload) -> Any:
        self.tbar.update()
        return payload


@dataclass
class Passthrough(DefaultResolveMixIn, interfaces.ITask):
    pass


@dataclass
class Project(interfaces.ITask):

    project: Callable[[interfaces.DocumentPayload], Any] = None

    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.ANY

    def process_payload(self, payload: interfaces.DocumentPayload) -> Any:
        return self.project(payload)


@dataclass
class ToContent(interfaces.ITask):
    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = Any

    def process_payload(self, payload: interfaces.DocumentPayload) -> Any:
        return payload.content


@dataclass
class ToDocumentContentTuple(interfaces.ITask):
    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.DOCUMENT_CONTENT_TUPLE

    def process_payload(self, payload: interfaces.DocumentPayload) -> Any:
        return payload.update(
            self.out_content_type,
            content=(
                payload.filename,
                payload.content,
            ),
        )


@dataclass
class Checkpoint(DefaultResolveMixIn, interfaces.ITask):

    filename: str = None

    def setup(self):
        super().setup()
        self.pipeline.put("checkpoint_file", self.filename)
        return self

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS, ContentType.TAGGEDFRAME]
        self.out_content_type = ContentType.PASSTHROUGH

    def outstream(self) -> Iterable[interfaces.DocumentPayload]:
        if os.path.isfile(self.filename):
            checkpoint_data: checkpoint.CheckpointData = checkpoint.load_checkpoint(
                self.filename, document_index_key_column=self.pipeline.payload.document_index_key
            )
            self.pipeline.payload.primary_document_index = checkpoint_data.document_index
            self.out_content_type = checkpoint_data.content_type
            payload_stream = checkpoint_data.payload_stream
        else:
            prior_content_type = self.pipeline.get_prior_content_type(self)
            if prior_content_type == ContentType.NONE:
                raise interfaces.PipelineError(
                    "Checkpoint file removed OR pipeline setup error. Checkpoint file does not exist AND checkpoint task has no prior task"
                )
            self.out_content_type = prior_content_type
            payload_stream = checkpoint.store_checkpoint(
                options=checkpoint.ContentSerializeOpts(
                    content_type_code=int(self.out_content_type),
                    as_binary=False,  # should be True if ContentType.SPARV_XML
                ),
                target_filename=self.filename,
                document_index=self.document_index,
                payload_stream=self.instream,
            )
        for payload in payload_stream:
            yield payload


@dataclass
class SaveTaggedFrame(DefaultResolveMixIn, interfaces.ITask):
    """Stores sequence of data frame documents. """

    filename: str = None

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGEDFRAME
        self.out_content_type = ContentType.TAGGEDFRAME

    def outstream(self) -> Iterable[interfaces.DocumentPayload]:
        for payload in checkpoint.store_checkpoint(
            options=checkpoint.ContentSerializeOpts(content_type_code=int(ContentType.TAGGEDFRAME)),
            target_filename=self.filename,
            document_index=self.document_index,
            payload_stream=self.instream,
        ):
            yield payload


@dataclass
class LoadTaggedFrame(DefaultResolveMixIn, interfaces.ITask):
    """Extracts text from payload.content based on annotations etc. """

    filename: str = None
    document_index_name: str = field(default="document_index.csv")

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TAGGEDFRAME

    def outstream(self) -> Iterable[interfaces.DocumentPayload]:

        checkpoint_data: checkpoint.CheckpointData = checkpoint.load_checkpoint(
            self.filename,
            document_index_key_column=self.pipeline.payload.document_index_key,
        )
        self.pipeline.payload.primary_document_index = checkpoint_data.document_index

        for payload in checkpoint_data.payload_stream:
            yield payload


@dataclass
class TextToTokens(interfaces.ITask):
    """Extracts tokens from payload.content, optinally transforming"""

    tokenize: Callable[[str], List[str]] = None
    text_transform_opts: TokensTransformOpts = None

    tokens_transform_opts: Optional[TokensTransformOpts] = None
    transformer: Optional[TokensTransformer] = None

    _text_transformer: TextTransformer = field(init=False)

    def setup(self):
        super().setup()
        self.pipeline.put("text_transform_opts", self.text_transform_opts)
        self.pipeline.put("tokens_transform_opts_text", self.tokens_transform_opts)
        return self

    def __post_init__(self):

        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS]
        self.out_content_type = ContentType.TOKENS
        self.tokenize = self.tokenize or default_tokenizer

        if self.text_transform_opts is not None:
            self._text_transformer = TextTransformer(text_transform_opts=self.text_transform_opts)

        if self.tokens_transform_opts is not None:
            if self.transformer is None:
                self.transformer = TokensTransformer(tokens_transform_opts=self.tokens_transform_opts)
            self.transformer.ingest(self.tokens_transform_opts)

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        if self.in_content_type == ContentType.TOKENS:
            tokens = payload.content
        else:
            if self._text_transformer is not None:
                self.tokenize(self._text_transformer.transform(payload.content))
            tokens = self.tokenize(payload.content)
        if self.transformer is not None:
            tokens = self.transformer.transform(tokens)
        return payload.update(self.out_content_type, tokens)


@dataclass
class TokensTransform(interfaces.ITask):
    """Transforms tokens payload.content"""

    tokens_transform_opts: TokensTransformOpts = None
    transformer: TokensTransformer = None

    def setup(self):
        super().setup()
        self.pipeline.put("tokens_transform_opts", self.tokens_transform_opts)
        return self

    def __post_init__(self):
        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.TOKENS
        self.transformer = TokensTransformer(tokens_transform_opts=self.tokens_transform_opts)

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(self.out_content_type, self.transformer.transform(payload.content))

    def add(self, transform: Callable[[List[str]], List[str]]) -> "TokensTransform":
        self.transformer.add(transform)
        return self


@dataclass
class TokensToText(interfaces.ITask):
    """Extracts text from payload.content"""

    def __post_init__(self):
        self.in_content_type = [ContentType.TOKENS, ContentType.TEXT]
        self.out_content_type = ContentType.TEXT

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(self.out_content_type, to_text(payload.content))


@dataclass
class TextToDTM(interfaces.ITask):
    def __post_init__(self):
        self.in_content_type = ContentType.DOCUMENT_CONTENT_TUPLE
        self.out_content_type = ContentType.VECTORIZED_CORPUS

    vectorize_opts: VectorizeOpts = None

    def setup(self):
        super().setup()
        self.pipeline.put("vectorize_opts", self.vectorize_opts)
        return self

    def outstream(self) -> VectorizedCorpus:
        corpus = convert.to_vectorized_corpus(
            stream=self.instream,
            vectorize_opts=self.vectorize_opts,
            document_index=self.pipeline.payload.document_index,
        )
        yield interfaces.DocumentPayload(content_type=ContentType.VECTORIZED_CORPUS, content=corpus)

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return None


@dataclass
class ChunkTokens(interfaces.ITask):
    chunk_size: int = None

    def setup(self):
        super().setup()
        self.pipeline.put("chunk_size", self.chunk_size)
        return self

    def __post_init__(self):
        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.TOKENS

    def outstream(self) -> Iterable[interfaces.DocumentPayload]:

        for payload in self.instream:
            tokens = payload.content
            if len(payload.content) < self.chunk_size:
                yield payload
            else:
                for chunk_id, i in enumerate(range(0, len(tokens), self.chunk_size)):
                    yield interfaces.DocumentPayload(
                        filename=payload.filename,
                        content_type=ContentType.TOKENS,
                        content=tokens[i : i + self.chunk_size],
                        chunk_id=chunk_id,
                    )


# @dataclass
# class TokensToTokenizedCorpus(interfaces.ITask):
#     def __post_init__(self):
#         self.in_content_type = [ContentType.DATAFRAME]
#         self.out_content_type = ContentType.TOKENIZED_CORPUS

#     tokens_transform_opts: TokensTransformOpts = None

#     def outstream(self) -> ITokenizedCorpus:

#         reader: ICorpusReader = TextReader()
#         tokenized_corpus = TokenizedCorpus(
#             reader=reader,
#             tokens_transform_opts=self.tokens_transform_opts,
#         )
#         return tokenized_corpus


# class TCorpus(ITokenizedCorpus):
#     """Krav: Reitererbar """

#     def __init__(self, pipeline: CorpusPipeline):
#         if pipeline.tasks[-1].content_type != ContentType.TOKENS:
#             raise interfaces.PipelineError("expected token stream")
#         self.pipeline = pipeline
#         # self.checkpoint = pipeline.

#     @property
#     def terms(self) -> Iterator[Iterator[str]]:
#         for payload in self.pipeline:
#             yield payload.content

#     @property
#     def metadata(self) -> List[Dict[str, Any]]:
#         return None

#     @property
#     def filenames(self) -> List[str]:
#         return None

#     @property
#     def documents(self) -> pd.DataFrame:
#         return self.pipeline.payload.document_index
