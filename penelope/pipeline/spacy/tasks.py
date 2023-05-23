from __future__ import annotations

from dataclasses import dataclass, field

from .. import interfaces
from .tagger import SpacyTagger


@dataclass
class ToSpacyDoc(interfaces.ITask):
    tagger: SpacyTagger = field(default=None)

    def __post_init__(self):
        self.in_content_type = [interfaces.ContentType.TEXT, interfaces.ContentType.TOKENS]
        self.out_content_type = interfaces.ContentType.SPACYDOC

    def setup(self):
        super().setup()
        self.pipeline.put("current_tagger", self.tagger, overwrite=False)
        return self

    def process_payload(self, payload: interfaces.DocumentPayload) -> interfaces.DocumentPayload:
        spacy_doc = self.tagger.to_document(payload.content)
        return payload.update(self.out_content_type, spacy_doc)
