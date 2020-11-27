from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from penelope.corpus import VectorizeOpts
from penelope.corpus.readers import ExtractTokensOpts2, TextReaderOpts, TextTransformOpts
from spacy.language import Language

from . import tasks

if TYPE_CHECKING:
    from . import pipeline


class PipelineShortcutMixIn:
    def load(
        self, *, reader_opts: TextReaderOpts = None, transform_opts: TextTransformOpts = None, source=None
    ) -> pipeline.CorpusPipeline:
        return self.add(tasks.LoadText(source=source, reader_opts=reader_opts, transform_opts=transform_opts))

    def dataframe_to_tokens(self, extract_tokens_opts: ExtractTokensOpts2) -> pipeline.CorpusPipeline:
        return self.add(tasks.DataFrameToTokens(extract_word_opts=extract_tokens_opts))

    def save_dataframe(self, filename: str) -> pipeline.CorpusPipeline:
        return self.add(tasks.SaveDataFrame(filename=filename))

    def load_dataframe(self, filename: str) -> pipeline.CorpusPipeline:
        """ _ => DATAFRAME """
        return self.add(tasks.LoadDataFrame(filename=filename))

    def checkpoint_dataframe(self, filename: str) -> pipeline.CorpusPipeline:
        """ DATAFRAME => [CHECKPOINT] => DATAFRAME """
        return self.add(tasks.CheckpointDataFrame(filename=filename))

    def tokens_to_text(self) -> pipeline.CorpusPipeline:
        """ [TOKEN] => TEXT """
        return self.add(tasks.TokensToText())

    def to_dtm(self, vectorize_opts: VectorizeOpts = None) -> pipeline.CorpusPipeline:
        """ TEXT => DTM """
        return self.add(tasks.TextToDTM(vectorize_opts=vectorize_opts or VectorizeOpts()))

    def to_content(self) -> pipeline.CorpusPipeline:
        return self.add(tasks.ToContent())

    def tqdm(self) -> pipeline.CorpusPipeline:
        return self.add(tasks.Tqdm())

    def passthrough(self) -> pipeline.CorpusPipeline:
        return self.add(tasks.Passthrough())


class SpacyPipelineShortcutMixIn:
    def text_to_spacy(self, nlp: Language = None) -> pipeline.CorpusPipeline:
        return self.add(tasks.TextToSpacy(nlp=nlp or self.nlp))

    def spacy_to_dataframe(self, attributes: List[str] = None) -> pipeline.CorpusPipeline:
        return self.add(tasks.SpacyToDataFrame(attributes=attributes))

    def spacy_to_pos_dataframe(self) -> pipeline.CorpusPipeline:
        return self.add(tasks.SpacyToDataFrame(attributes=['text', 'lemma_', 'pos_']))

    def set_spacy_model(self, language: Union[str, Language]) -> pipeline.CorpusPipeline:
        return self.add(tasks.SetSpacyModel(language=language))
