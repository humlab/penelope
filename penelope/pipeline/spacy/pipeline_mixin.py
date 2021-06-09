from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from . import tasks

if TYPE_CHECKING:
    from spacy.language import Language

    from ..pipelines import CorpusPipeline


class PipelineShortcutMixIn:
    def text_to_spacy(self: CorpusPipeline) -> CorpusPipeline:
        return self.add(tasks.ToSpacyDoc())

    def spacy_to_tagged_frame(self: CorpusPipeline, attributes: List[str] = None) -> CorpusPipeline:
        return self.add(tasks.SpacyDocToTaggedFrame(attributes=attributes))

    def spacy_to_pos_tagged_frame(self: CorpusPipeline) -> CorpusPipeline:
        return self.add(
            tasks.SpacyDocToTaggedFrame(
                attributes=['text', 'lemma_', 'pos_', 'is_punct', 'is_stop'],
            ),
        )

    def text_to_spacy_to_tagged_frame(self: CorpusPipeline) -> CorpusPipeline:
        return self.add(
            tasks.ToSpacyDocToTaggedFrame(
                attributes=['text', 'lemma_', 'pos_', 'is_punct', 'is_stop'],
            ),
        )

    def set_spacy_model(self: CorpusPipeline, language: Union[str, Language]) -> CorpusPipeline:
        return self.add(tasks.SetSpacyModel(lang_or_nlp=language))
