from dataclasses import dataclass
from typing import Any, Dict, List, Union

import spacy
from penelope.corpus.readers import ExtractTaggedTokensOpts, TaggedTokensFilterOpts
from spacy.language import Language

from .. import interfaces
from ..interfaces import ContentType, PipelineError
from ..tasks import DefaultResolveMixIn
from . import convert

DEFAULT_SPACY_DISABLES = ['vectors', 'textcat', 'dep', 'ner']


@dataclass
class SetSpacyModel(DefaultResolveMixIn, interfaces.ITask):
    """Extracts text from payload.content"""

    def __post_init__(self):
        self.in_content_type = ContentType.ANY
        self.out_content_type = ContentType.ANY

    lang_or_nlp: Union[str, Language] = None
    disables: List[str] = None

    def setup(self):
        disables = DEFAULT_SPACY_DISABLES if self.disables is None else self.disables
        nlp: Language = (
            spacy.load(self.lang_or_nlp, disable=disables) if isinstance(self.lang_or_nlp, str) else self.lang_or_nlp
        )
        self.pipeline.put("spacy_nlp", nlp)
        self.pipeline.put("disables", self.disables)
        return self


@dataclass
class ToSpacyDoc(interfaces.ITask):

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
class ToSpacyDocToTaggedFrame(interfaces.ITask):

    attributes: List[str] = None
    attribute_value_filters: Dict[str, Any] = None

    def setup(self):
        self.pipeline.put("spacy_attributes", self.attributes)
        return self

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS]
        self.out_content_type = ContentType.TAGGEDFRAME

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(
            self.out_content_type,
            convert.text_to_tagged_frame(
                document=payload.as_str(),
                attributes=self.attributes,
                attribute_value_filters=self.attribute_value_filters,
                nlp=self.pipeline.get("spacy_nlp"),
            ),
        )


@dataclass
class SpacyDocToTaggedFrame(interfaces.ITask):

    attributes: List[str] = None
    attribute_value_filters: Dict[str, Any] = None

    def __post_init__(self):
        self.in_content_type = ContentType.SPACYDOC
        self.out_content_type = ContentType.TAGGEDFRAME

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        return payload.update(
            self.out_content_type,
            convert.spacy_doc_to_tagged_frame(
                spacy_doc=payload.content,
                attributes=self.attributes,
                attribute_value_filters=self.attribute_value_filters,
            ),
        )


@dataclass
class TaggedFrameToTokens(interfaces.ITask):
    """Extracts text from payload.content based on annotations etc. """

    extract_opts: ExtractTaggedTokensOpts = None
    filter_opts: TaggedTokensFilterOpts = None

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGEDFRAME
        self.out_content_type = ContentType.TOKENS

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:

        return payload.update(
            self.out_content_type,
            convert.tagged_frame_to_tokens(
                payload.content,
                self.extract_opts,
                self.filter_opts,
            ),
        )
