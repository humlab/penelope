from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Union

from penelope.type_alias import TaggedFrame
from penelope.vendor import spacy_api

from .. import interfaces
from . import convert

DEFAULT_SPACY_DISABLES = ['textcat', 'parser']
DEFAULT_SPACY_EXCLUDES = ['ner']


@dataclass
class SpacyTagger(interfaces.IDocumentTagger):
    """_summary_

    Args:
        interfaces (_type_): _description_

    Returns:
        _type_: _description_
    """

    model: Union[str, spacy_api.Language] = None

    disable: list[str] = None
    exclude: list[str] = None
    keep_hyphens: bool = False
    remove_whitespace_ents: bool = False

    attributes: list[str] = None
    filters: dict[str, Any] = None

    def __post_init__(self):
        if self.disable is None:
            self.disable = DEFAULT_SPACY_DISABLES
        if self.exclude is None:
            self.exclude = DEFAULT_SPACY_EXCLUDES

    @cached_property
    def nlp(self) -> spacy_api.Language:
        if self.model is None:
            return None

        model: str | spacy_api.Language = spacy_api.prepend_spacy_path(self.model)

        return spacy_api.load_model(
            model=model,
            disable=self.disable,
            exclude=self.exclude,
            keep_hyphens=self.keep_hyphens,
            remove_whitespace_ents=self.remove_whitespace_ents,
        )

    def to_document(self, document: str | list[str] | spacy_api.Doc, disable: list[str] = None) -> spacy_api.Doc:
        if isinstance(document, spacy_api.Doc):
            return document

        if self.nlp is None:
            return None

        return self.nlp(self.to_text(document), disable=disable or self.disable)  # pylint: disable=not-callable

    def to_tagged_frame(
        self,
        document: spacy_api.Doc,
        attributes: list[str] = None,
        filters: dict[str, Any] = None,
    ) -> spacy_api.Doc:
        spacy_doc: spacy_api.Doc = self.to_document(document)
        return convert.spacy_doc_to_tagged_frame(
            spacy_doc=spacy_doc,
            attributes=attributes or self.attributes,
            attribute_value_filters=filters or self.filters,
        )

    def to_text(self, document: str | list[str]) -> str:
        if isinstance(document, (list, tuple)):
            return ' '.join(document)
        return document if document is not None else ""

    def tag(self, document: str | list[str] | spacy_api.Doc) -> TaggedFrame:
        """Implements IDocumentTagger interface"""
        return self.to_tagged_frame(document=document)
