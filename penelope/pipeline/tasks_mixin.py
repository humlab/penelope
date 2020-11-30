from __future__ import annotations

from typing import TYPE_CHECKING

from penelope.corpus import VectorizeOpts
from penelope.corpus.readers import TextReaderOpts, TextTransformOpts

from . import tasks

if TYPE_CHECKING:
    from . import pipelines

# FIXME:  pipelines.CorpusPipeline => pipeline.T_self
class PipelineShortcutMixIn:
    def load_text(
        self: pipelines.CorpusPipeline,
        *,
        reader_opts: TextReaderOpts = None,
        transform_opts: TextTransformOpts = None,
        source=None,
    ) -> pipelines.CorpusPipeline:
        return self.add(tasks.LoadText(source=source, reader_opts=reader_opts, transform_opts=transform_opts))

    def save_dataframe(self: pipelines.CorpusPipeline, filename: str) -> pipelines.CorpusPipeline:
        return self.add(tasks.SaveTaggedFrame(filename=filename))

    def load_dataframe(self: pipelines.CorpusPipeline, filename: str) -> pipelines.CorpusPipeline:
        """ _ => DATAFRAME """
        return self.add(tasks.LoadTaggedFrame(filename=filename))

    def checkpoint(self: pipelines.CorpusPipeline, filename: str) -> pipelines.CorpusPipeline:
        """ [DATAFRAME,TEXT,TOKENS] => [CHECKPOINT] => PASSTHROUGH """
        return self.add(tasks.Checkpoint(filename=filename))

    def tokens_to_text(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        """ [TOKEN] => TEXT """
        return self.add(tasks.TokensToText())

    def to_dtm(self: pipelines.CorpusPipeline, vectorize_opts: VectorizeOpts = None) -> pipelines.CorpusPipeline:
        """ (filename, TEXT => DTM) """
        return self.add(tasks.TextToDTM(vectorize_opts=vectorize_opts or VectorizeOpts()))

    def to_content(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.ToContent())

    def tqdm(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.Tqdm())

    def passthrough(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.Passthrough())

    def to_document_content_tuple(self) -> pipelines.CorpusPipeline:
        return self.add(tasks.ToDocumentContentTuple())
