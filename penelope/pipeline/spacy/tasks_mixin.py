from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from . import tasks

if TYPE_CHECKING:
    # from ..pipeline import T_self
    from spacy.language import Language

    from .pipelines import SpacyPipeline


class SpacyPipelineShortcutMixIn:
    def text_to_spacy(self: SpacyPipeline) -> SpacyPipeline:
        return self.add(tasks.ToSpacyDoc())

    def spacy_to_tagged_frame(self: SpacyPipeline, attributes: List[str] = None) -> SpacyPipeline:
        return self.add(tasks.SpacyDocToTaggedFrame(attributes=attributes))

    def spacy_to_pos_tagged_frame(self: SpacyPipeline) -> SpacyPipeline:
        return self.add(
            tasks.SpacyDocToTaggedFrame(
                attributes=['text', 'lemma_', 'pos_', 'is_space', 'is_punct'],
            ),
        )

    def set_spacy_model(self: SpacyPipeline, language: Union[str, Language]) -> SpacyPipeline:
        return self.add(tasks.SetSpacyModel(lang_or_nlp=language))
