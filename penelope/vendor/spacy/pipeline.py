import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, List, Mapping, Sequence, Union

import pandas as pd
from penelope.corpus import CorpusVectorizer, VectorizedCorpus, VectorizeOpts
from penelope.corpus.readers import TextReader
from penelope.corpus.readers.interfaces import TextSource
from spacy.language import Language

from .extract import ExtractTextOpts, dataframe_to_tokens, spacy_doc_to_annotated_dataframe, text_to_annotated_dataframe


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


@dataclass
class PipelinePayload:

    source: TextSource = None

    document_index_filename: pd.DataFrame = None
    document_index: pd.DataFrame = None

    # NOT USED: token2id: Mapping = None
    # NOT USED: annotation_opts: AnnotationOpts = None
    # NOT USED: tokens_transform_opts: TokensTransformOpts = None
    # NOT USED: extract_opts: Mapping = None


class ITask(abc.ABC):

    pipeline: "SpacyPipeline" = None
    instream: Iterable[DocumentPayload] = None

    def setup(self) -> "ITask":
        prior_task = self.pipeline.get_prior_to(self)
        if prior_task is not None:
            self.instream = prior_task.outstream()
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

    def resolve(self):
        """Resolves the pipeline by calling outstream on last task"""
        for task in self.tasks:
            task.setup()
        return self.tasks[-1].outstream()

    def add(self, task: Union[ITask, List[ITask]]) -> "SpacyPipeline":
        """Add one or more tasks to the pipeline. Hooks up a reference to pipeline for each task"""
        tasks = [task] if isinstance(task, ITask) else task
        self.tasks.extend(map(lambda x: x.hookup(self), tasks))
        return self

    def load(
        self,
        filename_pattern: str = "*.txt",
        filename_filter: List[str] = None,
        filename_fields: List[str] = None,
    ) -> "SpacyPipeline":
        return self.add(
            LoadText(
                filename_pattern=filename_pattern,
                filename_filter=filename_filter,
                filename_fields=filename_fields,
            )
        )

    def text_to_spacy(self, nlp: Language) -> "SpacyPipeline":
        return self.add(TextToSpacy(nlp=nlp))

    def spacy_to_dataframe(self, nlp: Language, attributes: List[str]) -> "SpacyPipeline":
        return self.add(SpacyToDataFrame(nlp=nlp, attributes=attributes))

    def dataframe_to_tokens(self, extract_text_opts: ExtractTextOpts) -> "SpacyPipeline":
        return self.add(DataFrameToTokens(extract_word_opts=extract_text_opts))

    def tokens_to_text(self) -> "SpacyPipeline":
        return self.add(TokensToText())

    def to_dtm(self, vectorize_opts: VectorizeOpts = None) -> "SpacyPipeline":
        return self.add(TextToDTM(vectorize_opts or VectorizeOpts()))


@dataclass
class LoadText(ITask):
    """Loads a text source into spaCy
    Also loads a document_index, and/or extracts value fields from filenames
    """

    filename_pattern: str = "*.txt"
    filename_filter: Union[List[str], Callable] = None
    filename_fields: List[str] = None

    def setup(self):

        text_reader: TextReader = (
            self.pipeline.payload.source
            if isinstance(self.pipeline.payload.source, TextReader)
            else TextReader(
                source=self.pipeline.payload.source,
                filename_pattern=self.filename_pattern,
                filename_filter=self.filename_filter,
                filename_fields=self.filename_fields,
            )
        )
        self.load_document_index(reader_index=text_reader.document_index)
        self.instream = (
            DocumentPayload(filename=filename, content_type=ContentType.TEXT, content=text)
            for filename, text in text_reader
        )

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
class TextToSpacy(ITask):

    nlp: Language = None

    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:
        return payload.update(ContentType.SPACYDOC, self.nlp(payload.content))


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

    extract_word_opts: ExtractTextOpts = None

    def _resolve(self, payload: DocumentPayload) -> DocumentPayload:

        return payload.update(ContentType.TOKENS, dataframe_to_tokens(payload.content, self.extract_word_opts))


def to_text(data: Union[str, Iterable[str]]):
    return data if isinstance(data, str) else ' '.join(data)


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
