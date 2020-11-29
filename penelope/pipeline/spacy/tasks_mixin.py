from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from penelope.corpus.readers.interfaces import SpacyExtractTokensOpts

from . import tasks

if TYPE_CHECKING:
    # from ..pipeline import T_self
    from spacy.language import Language

    from ..pipelines import SpacyPipeline


class SpacyPipelineShortcutMixIn:
    def text_to_spacy(self: SpacyPipeline) -> SpacyPipeline:
        return self.add(tasks.ToSpacyDoc())

    def spacy_to_dataframe(self: SpacyPipeline, attributes: List[str] = None) -> SpacyPipeline:
        return self.add(tasks.SpacyDocToTaggedFrame(attributes=attributes))

    def spacy_to_pos_dataframe(self: SpacyPipeline) -> SpacyPipeline:
        return self.add(tasks.SpacyDocToTaggedFrame(attributes=['text', 'lemma_', 'pos_']))

    def set_spacy_model(self: SpacyPipeline, language: Union[str, Language]) -> SpacyPipeline:
        return self.add(tasks.SetSpacyModel(lang_or_nlp=language))

    def dataframe_to_tokens(self: SpacyPipeline, extract_tokens_opts: SpacyExtractTokensOpts) -> SpacyPipeline:
        return self.add(tasks.TaggedFrameToTokens(extract_word_opts=extract_tokens_opts))
