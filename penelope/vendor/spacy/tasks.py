import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Union

import pandas as pd
import spacy
from penelope.corpus import CorpusVectorizer, VectorizedCorpus, VectorizeOpts
from penelope.corpus.readers import TextReader, TextReaderOpts, TextTransformOpts
from penelope.corpus.readers.interfaces import ExtractTokensOpts2
from penelope.vendor.spacy.convert import (
    dataframe_to_tokens,
    spacy_doc_to_annotated_dataframe,
    text_to_annotated_dataframe,
)
from spacy.language import Language
from tqdm.std import tqdm

from . import interfaces
from ._utils import to_text


class DefaultResolveMixIn:
    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload


@dataclass
class LoadText(DefaultResolveMixIn, interfaces.ITask):
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
        self.pipeline.payload.consolidate_document_index(text_reader.document_index)

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
        self.tbar = tqdm(total=len(self.document_index))

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


@dataclass
class TextToSpacy(interfaces.ITask):

    nlp: Language = None
    disable: List[str] = None

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        disable = self.disable or ['vectors', 'textcat', 'ner', 'parser']
        return payload.update(interfaces.ContentType.SPACYDOC, self.nlp(payload.content, disable=disable))


@dataclass
class TextToSpacyToDataFrame(interfaces.ITask):

    nlp: Language = None
    attributes: List[str] = None

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(
            interfaces.ContentType.DATAFRAME, text_to_annotated_dataframe(payload.content, self.attributes, self.nlp)
        )


@dataclass
class SpacyToDataFrame(interfaces.ITask):

    nlp: Language = None
    attributes: List[str] = None

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(
            interfaces.ContentType.DATAFRAME, spacy_doc_to_annotated_dataframe(payload.content, self.attributes)
        )


@dataclass
class DataFrameToTokens(interfaces.ITask):
    """Extracts text from payload.content based on annotations etc. """

    extract_word_opts: ExtractTokensOpts2 = None

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:

        return payload.update(
            interfaces.ContentType.TOKENS, dataframe_to_tokens(payload.content, self.extract_word_opts)
        )


@dataclass
class SaveDataFrame(DefaultResolveMixIn, interfaces.ITask):
    """Stores sequence of data frame documents. """

    filename: str = None

    def outstream(self) -> Iterable[interfaces.DocumentPayload]:
        for payload in self.pipeline.store_data_frame_outstream(self.filename, self.document_index, self.instream):
            yield payload


@dataclass
class LoadDataFrame(DefaultResolveMixIn, interfaces.ITask):
    """Extracts text from payload.content based on annotations etc. """

    filename: str = None
    document_index_name: str = field(default="document_index.csv")

    def outstream(self) -> Iterable[interfaces.DocumentPayload]:
        for payload in self.pipeline.load_data_frame_instream(self.filename):
            yield payload


@dataclass
class CheckpointDataFrame(DefaultResolveMixIn, interfaces.ITask):

    filename: str = None

    def outstream(self) -> Iterable[interfaces.DocumentPayload]:
        stream = (
            self.pipeline.load_data_frame_instream(self.filename)
            if os.path.isfile(self.filename)
            else self.pipeline.store_data_frame_outstream(self.filename, self.document_index, self.instream)
        )
        for payload in stream:
            yield payload


@dataclass
class SpacyModel(interfaces.ITask):
    """Extracts text from payload.content"""

    language: Union[str, Language]
    disables: List[str] = field(default_factory=list)

    def setup(self):
        disables = ['vectors', 'textcat', 'dep', 'ner'] if self.disables is None else self.disables
        nlp: Language = spacy.load(self.language, disable=disables) if isinstance(self.language, str) else Language
        self.pipeline.payload.put("spacy_nlp", nlp)

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(interfaces.ContentType.TEXT, to_text(payload.content))


@dataclass
class TokensToText(interfaces.ITask):
    """Extracts text from payload.content"""

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(interfaces.ContentType.TEXT, to_text(payload.content))


@dataclass
class TextToDTM(interfaces.ITask):
    """Extracts text from payload.content based on annotations etc. """

    vectorize_opts: VectorizeOpts = None

    def outstream(self) -> VectorizedCorpus:
        vectorizer = CorpusVectorizer()
        terms = (to_text(payload.content) for payload in self.instream)
        corpus = vectorizer.fit_transform_(
            terms, documents=self.pipeline.payload.document_index, vectorize_opts=self.vectorize_opts
        )
        return corpus

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return None
