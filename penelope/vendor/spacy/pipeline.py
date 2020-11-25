import abc
import collections
import os
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, List, Mapping, Sequence, Union

import pandas as pd
from penelope.corpus import CorpusVectorizer, VectorizedCorpus, VectorizeOpts
from penelope.corpus.readers import ExtractTokensOpts2, TextReader, TextReaderOpts, TextSource, TextTransformOpts
from penelope.utility import replace_extension
from spacy.language import Language
from tqdm import tqdm

from ._utils import read_data_frame_from_zip, to_text, write_data_frame_to_zip
from .convert import dataframe_to_tokens, spacy_doc_to_annotated_dataframe, text_to_annotated_dataframe


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

    document_index_filename: pd.DataFrame = None
    document_index: pd.DataFrame = None

    # NOT USED: token2id: Mapping = None
    # NOT USED: extract_tokens_opts: ExtractTokensOpts = None
    # NOT USED: tokens_transform_opts: TokensTransformOpts = None
    # NOT USED: extract_opts: Mapping = None


class ITask(abc.ABC):

    pipeline: "SpacyPipeline" = None
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
            yield self._resolve(payload)

    @abc.abstractmethod
    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:
        return payload

    def hookup(self, pipeline: "SpacyPipeline") -> "SpacyPipeline":
        self.pipeline = pipeline
        return self

    @property
    def document_index(self) -> pd.DataFrame:
        return self.pipeline.payload.document_index


class SpacyPipeline:
    def __init__(
        self,
        *,
        payload: PipelinePayload,
        tasks: Sequence[ITask] = None,
    ):
        self.payload = payload
        self.tasks: List[ITask] = []
        self.add(tasks or [])
        self.resolved = False

    # def process(self) -> Any:
    #     stream = self.payload.source
    #     for task in self.tasks:
    #         stream = task.resolve(pipeline=self, instream=stream)
    #     return stream

    def get_prior_to(self, task: ITask) -> ITask:
        index: int = self.tasks.index(task)
        if index > 0:
            return self.tasks[index - 1]
        return None

    # def prior_to_is_of_content_type(self, task: ITask, content_type: ContentType) -> bool:
    #     prior : ITask = self.get_prior_to(task)
    #     return prior.

    def resolve(self) -> Iterator[DocumentPayload]:
        """Resolves the pipeline by calling outstream on last task"""
        if not self.resolved:
            for task in self.tasks:
                task.chain()
                task.setup()
            self.resolved = True
        return self.tasks[-1].outstream()

    def add(self, task: Union[ITask, List[ITask]]) -> "SpacyPipeline":
        """Add one or more tasks to the pipeline. Hooks up a reference to pipeline for each task"""
        tasks = [task] if isinstance(task, ITask) else task
        self.tasks.extend(map(lambda x: x.hookup(self), tasks))
        return self

    def load(self, reader_opts: TextReaderOpts, transform_opts: TextTransformOpts = None) -> "SpacyPipeline":
        return self.add(LoadText(reader_opts=reader_opts, transform_opts=transform_opts))

    def text_to_spacy(self, nlp: Language) -> "SpacyPipeline":
        return self.add(TextToSpacy(nlp=nlp))

    def spacy_to_dataframe(self, nlp: Language, attributes: List[str]) -> "SpacyPipeline":
        return self.add(SpacyToDataFrame(nlp=nlp, attributes=attributes))

    def spacy_to_pos_dataframe(self, nlp: Language) -> "SpacyPipeline":
        return self.add(SpacyToDataFrame(nlp=nlp, attributes=['text', 'lemma_', 'pos_']))

    def dataframe_to_tokens(self, extract_tokens_opts: ExtractTokensOpts2) -> "SpacyPipeline":
        return self.add(DataFrameToTokens(extract_word_opts=extract_tokens_opts))

    def save_dataframe(self, filename: str) -> "SpacyPipeline":
        return self.add(SaveDataFrame(filename=filename))

    def load_dataframe(self, filename: str) -> "SpacyPipeline":
        return self.add(LoadDataFrame(filename=filename))

    def checkpoint_dataframe(self, filename: str) -> "SpacyPipeline":
        return self.add(CheckpointDataFrame(filename=filename))

    def tokens_to_text(self) -> "SpacyPipeline":
        return self.add(TokensToText())

    def to_dtm(self, vectorize_opts: VectorizeOpts = None) -> "SpacyPipeline":
        return self.add(TextToDTM(vectorize_opts or VectorizeOpts()))

    def to_content(self) -> "SpacyPipeline":
        return self.add(ToContent())

    def tqdm(self) -> "SpacyPipeline":
        return self.add(Tqdm())

    def passthrough(self) -> "SpacyPipeline":
        return self.add(Passthrough())

    def exhaust(self) -> "SpacyPipeline":
        if self.resolved:
            raise PipelineError("cannot exhaust an already resolved pipeline")
        collections.deque(self.resolve(), maxlen=0)
        return self

    def load_data_frame_instream(self, source_filename: str) -> Iterable[DocumentPayload]:
        document_index_name = self.payload.document_index_filename
        self.payload.source = source_filename
        with zipfile.ZipFile(source_filename, mode="r") as zf:
            filenames = zf.namelist()
            if document_index_name in filenames:
                self.payload.document_index = read_data_frame_from_zip(zf, document_index_name)
                filenames.remove(document_index_name)
            for filename in filenames:
                df = read_data_frame_from_zip(zf, filename)
                payload = DocumentPayload(content_type=ContentType.DATAFRAME, content=df, filename=filename)
                yield payload

    @staticmethod
    def store_data_frame_outstream(
        target_filename: str, document_index: pd.DataFrame, payload_stream: Iterator[DocumentPayload]
    ):
        with zipfile.ZipFile(target_filename, mode="w", compresslevel=zipfile.ZIP_DEFLATED) as zf:
            write_data_frame_to_zip(document_index, "document_index.csv", zf)
            for payload in payload_stream:
                filename = replace_extension(payload.filename, ".csv")
                write_data_frame_to_zip(payload.content, filename, zf)
                yield payload


@dataclass
class LoadText(ITask):
    """Loads a text source into spaCy
    Also loads a document_index, and/or extracts value fields from filenames
    """

    reader_opts: TextReaderOpts = None
    transform_opts: TextTransformOpts = None

    def setup(self):
        super().setup()
        text_reader: TextReader = (
            self.pipeline.payload.source
            if isinstance(self.pipeline.payload.source, TextReader)
            else TextReader.create(
                source=self.pipeline.payload.source,
                reader_opts=self.reader_opts,
                transform_opts=(self.transform_opts or TextTransformOpts()),
            )
        )
        self.load_document_index(reader_index=text_reader.document_index)
        self.instream = (
            DocumentPayload(filename=filename, content_type=ContentType.TEXT, content=text)
            for filename, text in text_reader
        )
        return self

    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:
        return payload

    def load_document_index(self, reader_index: pd.DataFrame):
        """Loads a document index by combining any existing index with the reader created index"""
        index = self.pipeline.payload.document_index
        index_filename = self.pipeline.payload.document_index_filename

        if index is None:
            if index_filename:
                index = pd.read_csv(index_filename, sep='\t', index_col=0)

        if index is not None:
            columns = [x for x in reader_index.columns if x not in index.columns]
            index = index.merge(reader_index[columns], left_index=True, right_index=True, how='left')
        else:
            index = reader_index

        self.pipeline.payload.document_index = index


@dataclass
class Tqdm(ITask):

    tbar = None

    def setup(self):
        super().setup()
        self.tbar = tqdm(total=len(self.document_index))

    def _resolve(self, payload: DocumentPayload) -> Any:
        self.tbar.update()
        return payload


@dataclass
class Passthrough(ITask):
    def _resolve(self, payload: DocumentPayload) -> Any:
        return payload


@dataclass
class Project(ITask):

    project: Callable[[DocumentPayload], Any] = None

    def _resolve(self, payload: DocumentPayload) -> Any:
        return self.project(payload)


@dataclass
class ToContent(ITask):
    def _resolve(self, payload: DocumentPayload) -> Any:
        return payload.content


@dataclass
class TextToSpacy(ITask):

    nlp: Language = None
    disable: List[str] = None

    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:
        disable = self.disable or ['vectors', 'textcat', 'ner', 'parser']
        return payload.update(ContentType.SPACYDOC, self.nlp(payload.content, disable=disable))


@dataclass
class TextToSpacyToDataFrame(ITask):

    nlp: Language = None
    attributes: List[str] = None

    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:
        return payload.update(
            ContentType.DATAFRAME, text_to_annotated_dataframe(payload.content, self.attributes, self.nlp)
        )


@dataclass
class SpacyToDataFrame(ITask):

    nlp: Language = None
    attributes: List[str] = None

    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:
        return payload.update(ContentType.DATAFRAME, spacy_doc_to_annotated_dataframe(payload.content, self.attributes))


@dataclass
class DataFrameToTokens(ITask):
    """Extracts text from payload.content based on annotations etc. """

    extract_word_opts: ExtractTokensOpts2 = None

    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:

        return payload.update(ContentType.TOKENS, dataframe_to_tokens(payload.content, self.extract_word_opts))


@dataclass
class SaveDataFrame(ITask):
    """Stores sequence of data frame documents. """

    filename: str = None

    def _resolve(self, payload: DocumentPayload) -> Any:
        return payload

    def outstream(self) -> Iterable[DocumentPayload]:
        for payload in self.pipeline.store_data_frame_outstream(self.filename, self.document_index, self.instream):
            yield payload


@dataclass
class LoadDataFrame(ITask):
    """Extracts text from payload.content based on annotations etc. """

    filename: str = None
    document_index_name: str = field(default="document_index.csv")

    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:
        return payload

    def outstream(self) -> Iterable[DocumentPayload]:
        for payload in self.pipeline.load_data_frame_instream(self.filename):
            yield payload


@dataclass
class CheckpointDataFrame(ITask):

    filename: str = None

    def _resolve(self, payload: DocumentPayload) -> Any:
        return payload

    def outstream(self) -> Iterable[DocumentPayload]:
        stream = (
            self.pipeline.load_data_frame_instream(self.filename)
            if os.path.isfile(self.filename)
            else self.pipeline.store_data_frame_outstream(self.filename, self.document_index, self.instream)
        )
        for payload in stream:
            yield payload


@dataclass
class TokensToText(ITask):
    """Extracts text from payload.content"""

    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:
        return payload.update(ContentType.TEXT, to_text(payload.content))


@dataclass
class TextToDTM(ITask):
    """Extracts text from payload.content based on annotations etc. """

    vectorize_opts: VectorizeOpts = None

    def outstream(self) -> VectorizedCorpus:
        vectorizer = CorpusVectorizer()
        terms = (to_text(payload.content) for payload in self.instream)
        corpus = vectorizer.fit_transform_(
            terms, documents=self.pipeline.payload.document_index, vectorize_opts=self.vectorize_opts
        )
        return corpus

    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:
        return None


def extract_text_to_vectorized_corpus(
    source: TextSource,
    nlp: Language,
    *,
    reader_opts: TextReaderOpts,
    transform_opts: TextTransformOpts,
    extract_tokens_opts: ExtractTokensOpts2,
    vectorize_opts: VectorizeOpts,
    document_index: pd.DataFrame = None,
) -> VectorizedCorpus:
    payload = PipelinePayload(source=source, document_index=document_index)
    pipeline = (
        SpacyPipeline(payload=payload)
        .load(reader_opts=reader_opts, transform_opts=transform_opts)
        .text_to_spacy(nlp=nlp)
        .spacy_to_dataframe(nlp=nlp, attributes=['text', 'lemma_', 'pos_'])
        .dataframe_to_tokens(extract_tokens_opts=extract_tokens_opts)
        .tokens_to_text()
        .to_dtm(vectorize_opts)
    )

    corpus = pipeline.resolve()

    return corpus
