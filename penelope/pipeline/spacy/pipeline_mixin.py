from __future__ import annotations

from typing import TYPE_CHECKING

from . import tasks
from .tagger import SpacyTagger

if TYPE_CHECKING:
    from ..pipelines import CorpusPipeline

# pylint: disable=no-member


class PipelineShortcutMixIn:
    def text_to_spacy(self: CorpusPipeline, tagger: SpacyTagger) -> CorpusPipeline:
        return self.add(tasks.ToSpacyDoc(tagger=tagger))

    # def spacy_to_tagged_frame(
    #     self: CorpusPipeline, attributes: list[str] = None, filters: dict[str, Any] = None
    # ) -> CorpusPipeline:
    #     return self.add(tasks.SpacyDocToTaggedFrame(attributes=attributes, filters=filters))
