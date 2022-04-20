from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from penelope.type_alias import TaggedFrame
from penelope.vendor import spacy_api

from .. import interfaces
from ..tasks import DefaultResolveMixIn, ToTaggedFrame
from . import convert

DEFAULT_SPACY_DISABLES = ['vectors', 'textcat', 'dep', 'ner']


@dataclass
class SetSpacyModel(DefaultResolveMixIn, interfaces.ITask):
    """Extracts text from payload.content"""

    def __post_init__(self):
        self.in_content_type = interfaces.ContentType.ANY
        self.out_content_type = interfaces.ContentType.ANY

    name_or_nlp: Union[str, spacy_api.Language] = None

    disable: List[str] = None
    exclude: List[str] = None
    keep_hyphens: bool = False
    remove_whitespace_ents: bool = False

    def setup(self):
        self.disable = DEFAULT_SPACY_DISABLES if self.disable is None else self.disable
        self.name_or_nlp: spacy_api.Language = spacy_api.load_model(
            name_or_nlp=self.name_or_nlp,
            disable=self.disable,
            exclude=self.exclude,
            keep_hyphens=self.keep_hyphens,
            remove_whitespace_ents=self.remove_whitespace_ents,
        )
        self.pipeline.put("spacy_nlp", self.name_or_nlp)
        self.pipeline.put("disables", self.disable)
        return self


@dataclass
class ToSpacyDoc(interfaces.ITask):

    disable: List[str] = None

    def __post_init__(self):
        self.in_content_type = [interfaces.ContentType.TEXT, interfaces.ContentType.TOKENS]
        self.out_content_type = interfaces.ContentType.SPACYDOC

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        disable = self.disable or DEFAULT_SPACY_DISABLES
        nlp: spacy_api.Language = self.pipeline.get("spacy_nlp")
        if nlp is None:
            raise interfaces.PipelineError("spaCy.Language model not set (task SetSpacyModel)")
        content = self._get_content_as_text(payload)
        spacy_doc = nlp(content, disable=disable)
        return payload.update(self.out_content_type, spacy_doc)

    @staticmethod
    def _get_content_as_text(payload):
        if payload.content_type == interfaces.ContentType.TOKENS:
            return ' '.join(payload.content)
        return payload.content


@dataclass
class SpacyDocToTaggedFrame(ToTaggedFrame):
    def __post_init__(self):
        super().__post_init__()
        self.in_content_type = interfaces.ContentType.SPACYDOC
        self.tagger = self.spacy_tagger

    def spacy_tagger(
        self, payload: interfaces.DocumentPayload, attributes: List[str], attribute_value_filters: Dict[str, Any]
    ) -> TaggedFrame:
        return convert.spacy_doc_to_tagged_frame(
            spacy_doc=payload.content,
            attributes=attributes,
            attribute_value_filters=attribute_value_filters,
        )


@dataclass
class ToSpacyDocToTaggedFrame(ToTaggedFrame):
    def __post_init__(self):
        super().__post_init__()
        self.in_content_type = [interfaces.ContentType.TEXT, interfaces.ContentType.TOKENS]
        self.out_content_type = interfaces.ContentType.TAGGED_FRAME
        self.tagger = self.spacy_tagger

    def spacy_tagger(
        self, payload: interfaces.DocumentPayload, attributes: List[str], attribute_value_filters: Dict[str, Any]
    ) -> TaggedFrame:
        return convert.text_to_tagged_frame(
            document=payload.as_str(),
            attributes=attributes,
            attribute_value_filters=attribute_value_filters,
            nlp=self.pipeline.get("spacy_nlp"),
        )
