import itertools
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import pandas as pd
from penelope import co_occurrence, utility
from penelope.corpus import TokensTransformer, TokensTransformOpts, VectorizedCorpus, VectorizeOpts, default_tokenizer
from penelope.corpus.readers import (
    ExtractTaggedTokensOpts,
    TaggedTokensFilterOpts,
    TextReader,
    TextReaderOpts,
    TextSource,
    TextTransformer,
    TextTransformOpts,
)
from penelope.corpus.readers.tng.factory import create_sparv_xml_corpus_reader
from penelope.corpus.readers.tng.reader import CorpusReader
from tqdm.auto import tqdm

from . import checkpoint, convert
from .interfaces import ContentType, DocumentPayload, DocumentTagger, ITask, PipelineError
from .tasks_mixin import CountTokensMixIn, DefaultResolveMixIn


@dataclass
class LoadText(DefaultResolveMixIn, ITask):
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

    def setup(self) -> ITask:
        super().setup()

        self.transform_opts = self.transform_opts or TextTransformOpts()

        if self.source is not None:
            self.pipeline.payload.source = self.source

        text_reader: TextReader = TextReader.create(
            source=self.pipeline.payload.source,
            reader_opts=self.reader_opts,
            transform_opts=self.transform_opts,
        )

        self.pipeline.payload.set_reader_index(text_reader.document_index)
        self.pipeline.payload.metadata = text_reader.metadata
        self.pipeline.put("text_reader_opts", self.reader_opts.props)
        self.pipeline.put("text_transform_opts", self.transform_opts.props)

        self.instream = (
            DocumentPayload(filename=filename, content_type=ContentType.TEXT, content=text)
            for filename, text in text_reader
        )
        return self


@dataclass
class Tqdm(ITask):

    tbar = None

    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.ANY

    def setup(self) -> ITask:
        super().setup()
        self.tbar = tqdm(
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
    options: checkpoint.CorpusSerializeOpts = None

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("checkpoint_file", self.filename)
        return self

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS, ContentType.TAGGEDFRAME]
        self.out_content_type = ContentType.PASSTHROUGH

    def outstream(self) -> Iterable[DocumentPayload]:
        if os.path.isfile(self.filename):
            checkpoint_data: checkpoint.CheckpointData = checkpoint.load_checkpoint(self.filename, options=self.options)
            self.pipeline.payload.effective_document_index = checkpoint_data.document_index
            self.out_content_type = checkpoint_data.content_type
            payload_stream = checkpoint_data.payload_stream
        else:
            prior_content_type = self.pipeline.get_prior_content_type(self)
            if prior_content_type == ContentType.NONE:
                raise PipelineError(
                    "Checkpoint file removed OR pipeline setup error. Checkpoint file does not exist AND checkpoint task has no prior task"
                )
            self.out_content_type = prior_content_type
            options = (self.options or checkpoint.CorpusSerializeOpts()).as_type(self.out_content_type)
            payload_stream = checkpoint.store_checkpoint(
                options=options,
                target_filename=self.filename,
                document_index=self.document_index,
                payload_stream=self.instream,
            )
        for payload in payload_stream:
            yield payload


@dataclass
class SaveTaggedCSV(DefaultResolveMixIn, ITask):
    """Stores sequence of tagged data frame documents to archive. """

    filename: str = None
    options: checkpoint.CorpusSerializeOpts = None

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGEDFRAME
        self.out_content_type = ContentType.TAGGEDFRAME

    def outstream(self) -> Iterable[DocumentPayload]:
        options = (self.options or checkpoint.CorpusSerializeOpts()).as_type(ContentType.TAGGEDFRAME)
        for payload in checkpoint.store_checkpoint(
            options=options,
            target_filename=self.filename,
            document_index=self.document_index,
            payload_stream=self.instream,
        ):
            yield payload


@dataclass
class LoadTaggedCSV(CountTokensMixIn, DefaultResolveMixIn, ITask):
    """Loads CSV files stored in a ZIP as Pandas data frames. """

    filename: str = None
    options: checkpoint.CorpusSerializeOpts = None
    extra_reader_opts: TextReaderOpts = None  # Use if e.g. document index should be created

    checkpoint_data: checkpoint.CheckpointData = field(default=None, init=None, repr=None)

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TAGGEDFRAME

    def setup(self) -> ITask:
        super().setup()
        self.checkpoint_data = checkpoint.load_checkpoint(
            self.filename, options=self.options, reader_opts=self.extra_reader_opts
        )
        self.pipeline.payload.set_reader_index(self.checkpoint_data.document_index)
        if self.extra_reader_opts:
            self.pipeline.put("text_reader_opts", self.extra_reader_opts.props)

        self.instream = (payload for payload in self.checkpoint_data.payload_stream)

        return self

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        self.register_token_counts(payload)
        return payload


@dataclass
class LoadTaggedXML(CountTokensMixIn, DefaultResolveMixIn, ITask):
    """Loads Sparv export documents stored as individual XML files in a ZIP-archive into a Pandas data frames. """

    filename: str = None
    reader_opts: TextReaderOpts = None
    corpus_reader: CorpusReader = field(default=None, init=None, repr=None)

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TAGGEDFRAME

    def setup(self) -> ITask:
        super().setup()
        self.corpus_reader = create_sparv_xml_corpus_reader(
            source_path=self.filename or self.pipeline.payload.source,
            reader_opts=self.reader_opts or self.pipeline.config.text_reader_opts,
            sparv_version=int(self.pipeline.payload.get("sparv_version", 0)),
            content_type="pandas",
        )
        self.pipeline.payload.set_reader_index(self.corpus_reader.document_index)
        self.instream = (
            DocumentPayload(
                content_type=ContentType.TAGGEDFRAME,
                filename=document,
                content=content,
                filename_values=None,
            )
            for document, content in self.corpus_reader
        )

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        self.register_token_counts(payload)
        return payload


@dataclass
class TextToTokens(ITask):
    """Extracts tokens from payload.content, optinally transforming"""

    tokenize: Callable[[str], List[str]] = None
    text_transform_opts: TokensTransformOpts = None

    tokens_transform_opts: Optional[TokensTransformOpts] = None
    transformer: Optional[TokensTransformer] = None

    _text_transformer: TextTransformer = field(init=False)

    def setup(self) -> ITask:
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

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
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
class ToTaggedFrame(CountTokensMixIn, ITask):

    attributes: List[str] = None
    attribute_value_filters: Dict[str, Any] = None
    tagger: DocumentTagger = None

    def setup(self) -> ITask:
        self.pipeline.put("tagged_attributes", self.attributes)
        return self

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS]
        self.out_content_type = ContentType.TAGGEDFRAME

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

        tagged_frame: pd.DataFrame = self.tagger(
            payload=payload,
            attributes=self.attributes,
            attribute_value_filters=self.attribute_value_filters,
        )

        payload = payload.update(self.out_content_type, tagged_frame)

        self.register_token_counts(payload)

        return payload


@dataclass
class TaggedFrameToTokens(CountTokensMixIn, ITask):
    """Extracts text from payload.content based on annotations etc. """

    extract_opts: ExtractTaggedTokensOpts = None
    filter_opts: TaggedTokensFilterOpts = None

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGEDFRAME
        self.out_content_type = ContentType.TOKENS

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

        if self.pipeline.get('pos_column', None) is None:
            raise PipelineError("expected `pos_column` in `payload.memory_store` found None")

        tokens: Iterable[str] = convert.tagged_frame_to_tokens(
            doc=payload.content,
            extract_opts=self.extract_opts,
            filter_opts=self.filter_opts,
            **(self.pipeline.payload.tagged_columns_names or {}),
        )

        tokens = list(tokens)

        self.update_document_properties(payload, n_tokens=len(tokens))

        return payload.update(self.out_content_type, tokens)


@dataclass
class TokensTransform(ITask):
    """Transforms tokens payload.content"""

    tokens_transform_opts: TokensTransformOpts = None
    transformer: TokensTransformer = None

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("tokens_transform_opts", self.tokens_transform_opts)
        return self

    def __post_init__(self):
        self.in_content_type = ContentType.TOKENS
        self.out_content_type = ContentType.TOKENS
        self.transformer = TokensTransformer(tokens_transform_opts=self.tokens_transform_opts)

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        # FIXME Must update n_tokens
        return payload.update(self.out_content_type, self.transformer.transform(payload.content))

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

    def outstream(self) -> VectorizedCorpus:
        # FIXME: #30 [Bug] Index not set since pipeline is not exhaused at this point:
        corpus = convert.to_vectorized_corpus(
            stream=self.instream,
            vectorize_opts=self.vectorize_opts,
            document_index=lambda: self.pipeline.payload.document_index,
        )
        yield DocumentPayload(content_type=ContentType.VECTORIZED_CORPUS, content=corpus)

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None


@dataclass
class Vocabulary(ITask):

    token2id: Mapping[str, int] = None

    def __post_init__(self):
        self.in_content_type = [ContentType.TOKENS, ContentType.TAGGEDFRAME]
        self.out_content_type = ContentType.PASSTHROUGH

    def setup(self) -> ITask:
        self.token2id = defaultdict()
        self.token2id.default_factory = self.token2id.__len__
        self.pipeline.payload.token2id = self.token2id
        return self

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        for token in self.tokens_iter(payload):
            _ = self.token2id[token]
        return payload

    def tokens_iter(self, payload: DocumentPayload) -> Iterable[str]:

        if payload.content_type == ContentType.TOKENS:
            return payload.content

        tagged_frame: pd.DataFrame = payload.content
        column_names = self.pipeline.payload.tagged_columns_names
        return itertools.chain(
            tagged_frame[column_names['text_column']],
            tagged_frame[column_names['lemma_column']],
        )


@dataclass
class ToCoOccurrence(ITask):
    def __post_init__(self):
        self.in_content_type = [ContentType.DOCUMENT_CONTENT_TUPLE, ContentType.TOKENS]
        self.out_content_type = ContentType.CO_OCCURRENCE_DATAFRAME

    context_opts: co_occurrence.ContextOpts = None
    global_threshold_count: int = None
    partition_column: str = field(default='year')

    def setup(self) -> ITask:
        super().setup()
        self.pipeline.put("context_opts", self.context_opts)
        self.pipeline.put("global_threshold_count", self.global_threshold_count)
        self.pipeline.put("partition_column", self.partition_column)
        return self

    def outstream(self) -> VectorizedCorpus:

        # if self.pipeline.get_prior_content_type(self)  == ContentType.DOCUMENT_CONTENT_TUPLE:
        instream = (x.content for x in self.instream)
        # else:
        #     instream = ((x.filename, x.content) for x in self.instream)

        compute_result: co_occurrence.ComputeResult = co_occurrence.partitioned_corpus_co_occurrence(
            stream=instream,
            payload=self.pipeline.payload,
            context_opts=self.context_opts,
            global_threshold_count=self.global_threshold_count,
            partition_column=self.partition_column,
        )
        yield DocumentPayload(content_type=ContentType.CO_OCCURRENCE_DATAFRAME, content=compute_result)

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        return None


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

    def outstream(self) -> Iterable[DocumentPayload]:

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

    def outstream(self) -> Iterable[DocumentPayload]:
        self.abort()

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        self.abort()


# class Split(ITask):

#     partioner: Callable = None

#     def __post_init__(self):
#         self.in_content_type = ContentType.ANY
#         self.out_content_type = ContentType.STREAM

#     def setup(self) -> ITask:
#         super().setup()

#     def process_payload(self, payload: DocumentPayload) -> Any:
#         raise NotImplementedError()

#     def outstream(self) -> Iterable[DocumentPayload]:
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

#     def outstream(self) -> Iterable[DocumentPayload]:
#         raise NotImplementedError()
