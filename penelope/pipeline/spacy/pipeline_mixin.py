from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from . import tasks

if TYPE_CHECKING:
    from penelope.vendor.spacy_api import Language

    from ..pipelines import CorpusPipeline

# pylint: disable=no-member


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

    def set_spacy_model(
        self: CorpusPipeline,
        name_or_nlp: Union[str, Language],
        disable: List[str] = None,
        exclude: List[str] = None,
        keep_hyphens: bool = False,
        remove_whitespace_ents: bool = False,
    ) -> CorpusPipeline:
        return self.add(
            tasks.SetSpacyModel(
                name_or_nlp=name_or_nlp,
                disable=disable,
                exclude=exclude,
                keep_hyphens=keep_hyphens,
                remove_whitespace_ents=remove_whitespace_ents,
            )
        )
