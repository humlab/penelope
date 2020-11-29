from dataclasses import dataclass
from typing import List, Union

import spacy
from penelope.corpus.readers import SpacyExtractTokensOpts
from spacy.language import Language

from .. import interfaces
from ..interfaces import ContentType, PipelineError
from ..tasks import DefaultResolveMixIn
from . import convert

DEFAULT_SPACY_DISABLES = ['vectors', 'textcat', 'dep', 'ner']


@dataclass
class SetSpacyModel(DefaultResolveMixIn, interfaces.ITask):
    """Extracts text from payload.content"""

    lang_or_nlp: Union[str, Language] = None
    disables: List[str] = None

    def setup(self):
        disables = DEFAULT_SPACY_DISABLES if self.disables is None else self.disables
        nlp: Language = spacy.load(self.lang_or_nlp, disable=disables) if isinstance(self.lang_or_nlp, str) else self.lang_or_nlp
        self.pipeline.put("spacy_nlp", nlp)
        return self


@dataclass
class TextToSpacy(interfaces.ITask):

    disable: List[str] = None

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS]
        self.out_content_type = ContentType.SPACYDOC

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        disable = self.disable or DEFAULT_SPACY_DISABLES
        nlp: Language = self.pipeline.get("spacy_nlp")
        if nlp is None:
            raise PipelineError("spacy.Language model not set (task SetSpacyModel)")
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
                nlp=self.pipeline.get("spacy_nlp"),
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
