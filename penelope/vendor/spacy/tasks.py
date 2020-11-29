import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Union

import spacy
from penelope.corpus import VectorizedCorpus, VectorizeOpts
from penelope.corpus.readers import SpacyExtractTokensOpts, TextReader, TextReaderOpts, TextSource, TextTransformOpts
from spacy.language import Language
from tqdm.std import tqdm

from . import convert, interfaces
from ._utils import consolidate_document_index, to_text
from .interfaces import ContentType, PipelineError


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

    def setup(self):
        super().setup()
        if self.source is not None:
            self.pipeline.payload.source = self.source
        text_reader: TextReader = (
            self.pipeline.payload.source
            if isinstance(self.pipeline.payload.source, TextReader)
            else TextReader.create(
                source=self.pipeline.payload.source,
                reader_opts=self.reader_opts,
                transform_opts=(self.transform_opts or TextTransformOpts()),
            )
        )
        self.pipeline.payload.document_index = consolidate_document_index(
            index=self.pipeline.payload.document_index,
            reader_index=text_reader.document_index,
        )
        self.instream = (
            interfaces.DocumentPayload(filename=filename, content_type=interfaces.ContentType.TEXT, content=text)
            for filename, text in text_reader
        )
        return self


@dataclass
class Tqdm(interfaces.ITask):

    tbar = None

    def setup(self):
        super().setup()
        self.tbar = tqdm(total=len(self.document_index) if self.document_index is not None else None)
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

    def process_payload(self, payload: interfaces.DocumentPayload) -> Any:
        return self.project(payload)


@dataclass
class ToContent(interfaces.ITask):
    def process_payload(self, payload: interfaces.DocumentPayload) -> Any:
        return payload.content


DEFAULT_SPACY_DISABLES = ['vectors', 'textcat', 'dep', 'ner']


@dataclass
class SetSpacyModel(DefaultResolveMixIn, interfaces.ITask):
    """Extracts text from payload.content"""

    language: Union[str, Language] = None
    disables: List[str] = None

    def setup(self):
        disables = DEFAULT_SPACY_DISABLES if self.disables is None else self.disables
        nlp: Language = spacy.load(self.language, disable=disables) if isinstance(self.language, str) else Language
        self.pipeline.put("spacy_nlp", nlp)
        return self


@dataclass
class TextToSpacy(interfaces.ITask):

    nlp: Language = None
    disable: List[str] = None

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS]
        self.out_content_type = ContentType.SPACYDOC

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        disable = self.disable or DEFAULT_SPACY_DISABLES
        nlp = self.pipeline.get("spacy_nlp", self.nlp)
        content = self._get_content_as_text(payload)
        spacy_doc = nlp(content, disable=disable)
        return payload.update(self.out_content_type, spacy_doc)

    @staticmethod
    def _get_content_as_text(payload):
        if payload.content_type == ContentType.TOKENS:
            return ' '.join(payload.content)
        return payload.content


@dataclass
class TextToSpacyToDataFrame(interfaces.ITask):

    nlp: Language = None
    attributes: List[str] = None

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS]
        self.out_content_type = ContentType.DATAFRAME

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(
            self.out_content_type,
            convert.text_to_annotated_dataframe(
                document=payload.as_str(),
                attributes=self.attributes,
                nlp=self.pipeline.get("spacy_nlp", self.nlp),
            ),
        )


@dataclass
class SpacyToDataFrame(interfaces.ITask):

    attributes: List[str] = None

    def __post_init__(self):
        self.in_content_type = ContentType.SPACYDOC
        self.out_content_type = ContentType.DATAFRAME

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(
            self.out_content_type,
            convert.spacy_doc_to_annotated_dataframe(
                payload.content,
                self.attributes,
            ),
        )


@dataclass
class DataFrameToTokens(interfaces.ITask):
    """Extracts text from payload.content based on annotations etc. """

    extract_word_opts: SpacyExtractTokensOpts = None

    def __post_init__(self):
        self.in_content_type = ContentType.DATAFRAME
        self.out_content_type = ContentType.TOKENS

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:

        return payload.update(
            self.out_content_type,
            convert.dataframe_to_tokens(
                payload.content,
                self.extract_word_opts,
            ),
        )


@dataclass
class Checkpoint(DefaultResolveMixIn, interfaces.ITask):

    filename: str = None

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS, ContentType.DATAFRAME]
        self.out_content_type = ContentType.PASSTHROUGH

    def outstream(self) -> Iterable[interfaces.DocumentPayload]:
        if os.path.isfile(self.filename):
            checkpoint_data: convert.CheckpointData = convert.load_checkpoint(self.filename)
            self.pipeline.payload.document_index = checkpoint_data.document_index
            self.out_content_type = checkpoint_data.content_type
            payload_stream = checkpoint_data.payload_stream
        else:
            prior_content_type = self.pipeline.get_prior_content_type(self)
            if prior_content_type == ContentType.NONE:
                raise PipelineError(
                    "Checkpoint file removed OR pipeline setup error. Checkpoint file does not exist AND checkpoint task has no prior task"
                )
            self.out_content_type = prior_content_type
            payload_stream = convert.store_checkpoint(
                options=convert.ContentSerializeOpts(
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
class SaveDataFrame(DefaultResolveMixIn, interfaces.ITask):
    """Stores sequence of data frame documents. """

    filename: str = None

    def __post_init__(self):
        self.in_content_type = ContentType.DATAFRAME
        self.out_content_type = ContentType.DATAFRAME

    def outstream(self) -> Iterable[interfaces.DocumentPayload]:
        for payload in convert.store_checkpoint(
            options=convert.ContentSerializeOpts(content_type_code=int(ContentType.DATAFRAME)),
            target_filename=self.filename,
            document_index=self.document_index,
            payload_stream=self.instream,
        ):
            yield payload


@dataclass
class LoadDataFrame(DefaultResolveMixIn, interfaces.ITask):
    """Extracts text from payload.content based on annotations etc. """

    filename: str = None
    document_index_name: str = field(default="document_index.csv")

    def __post_init__(self):
        self.in_content_type = None
        self.out_content_type = ContentType.DATAFRAME

    def outstream(self) -> Iterable[interfaces.DocumentPayload]:

        checkpoint_data: convert.CheckpointData = convert.load_checkpoint(self.filename)
        self.pipeline.payload.document_index = checkpoint_data.document_index

        for payload in checkpoint_data.payload_stream:
            yield payload


@dataclass
class TokensToText(interfaces.ITask):
    """Extracts text from payload.content"""

    def __post_init__(self):
        self.in_content_type = [ContentType.TOKENS, ContentType.TEXT]
        self.out_content_type = ContentType.TEXT

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(self.out_content_type, to_text(payload.content))


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
#             raise PipelineError("expected token stream")
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


PartitionKeys = Union[str, List[str], Callable]


@dataclass
class TextToDTM(interfaces.ITask):
    def __post_init__(self):
        self.in_content_type = ContentType.TEXT
        self.out_content_type = ContentType.VECTORIZED_CORPUS

    vectorize_opts: VectorizeOpts = None

    def outstream(self) -> VectorizedCorpus:
        corpus = convert.to_vectorized_corpus(
            stream=self.instream,
            vectorize_opts=self.vectorize_opts,
            document_index=self.pipeline.payload.document_index,
        )
        return corpus

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return None
