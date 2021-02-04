from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from penelope import co_occurrence
from penelope.corpus import TokensTransformer, TokensTransformOpts, VectorizeOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts, TaggedTokensFilterOpts, TextReaderOpts, TextTransformOpts

from . import tasks
from .checkpoint import CorpusSerializeOpts

if TYPE_CHECKING:
    from . import pipelines

# FIXME:  pipelines.CorpusPipeline => pipeline.T_self
class PipelineShortcutMixIn:
    """Shortcuts for specific tasks that can be injected to derived pipelines"""

    def load_text(
        self: pipelines.CorpusPipeline,
        *,
        reader_opts: TextReaderOpts = None,
        transform_opts: TextTransformOpts = None,
        source=None,
    ) -> pipelines.CorpusPipeline:
        return self.add(tasks.LoadText(source=source, reader_opts=reader_opts, transform_opts=transform_opts))

    def save_tagged_frame(
        self: pipelines.CorpusPipeline, filename: str, options: CorpusSerializeOpts
    ) -> pipelines.CorpusPipeline:
        return self.add(tasks.SaveTaggedCSV(filename=filename, options=options))

    def load_tagged_frame(
        self: pipelines.CorpusPipeline,
        filename: str,
        options: CorpusSerializeOpts,
        extra_reader_opts: TextReaderOpts = None,
    ) -> pipelines.CorpusPipeline:
        """ _ => DATAFRAME """
        return self.add(tasks.LoadTaggedCSV(filename=filename, options=options, extra_reader_opts=extra_reader_opts))

    def load_tagged_xml(
        self: pipelines.CorpusPipeline, filename: str, options: CorpusSerializeOpts
    ) -> pipelines.CorpusPipeline:
        """ SparvXML => DATAFRAME """
        return self.add(tasks.LoadTaggedXML(filename=filename, options=options))

    def checkpoint(self: pipelines.CorpusPipeline, filename: str) -> pipelines.CorpusPipeline:
        """ [DATAFRAME,TEXT,TOKENS] => [CHECKPOINT] => PASSTHROUGH """
        return self.add(tasks.Checkpoint(filename=filename))

    def tokens_to_text(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        """ [TOKEN] => TEXT """
        return self.add(tasks.TokensToText())

    def text_to_tokens(
        self,
        *,
        text_transform_opts: TextTransformOpts,
        tokens_transform_opts: TokensTransformOpts = None,
        transformer: TokensTransformer = None,
    ) -> pipelines.CorpusPipeline:
        """ TOKEN => TOKENS """
        return self.add(
            tasks.TextToTokens(
                text_transform_opts=text_transform_opts,
                tokens_transform_opts=tokens_transform_opts,
                transformer=transformer,
            )
        )

    def tokens_transform(
        self, *, tokens_transform_opts: TokensTransformOpts, transformer: TokensTransformer = None
    ) -> pipelines.CorpusPipeline:
        """ TOKEN => TOKENS """
        return self.add(tasks.TokensTransform(tokens_transform_opts=tokens_transform_opts, transformer=transformer))

    def to_dtm(self: pipelines.CorpusPipeline, vectorize_opts: VectorizeOpts = None) -> pipelines.CorpusPipeline:
        """ (filename, TEXT => DTM) """
        return self.add(tasks.TextToDTM(vectorize_opts=vectorize_opts or VectorizeOpts()))

    def to_co_occurrence(
        self: pipelines.CorpusPipeline,
        context_opts: co_occurrence.ContextOpts = None,
        partition_column: str = 'year',
        global_threshold_count: int = None,
    ) -> pipelines.CorpusPipeline:
        """ (filename, DOCUMENT_CONTENT_TUPLES => DATAFRAME) """
        return self.add(
            tasks.ToCoOccurrence(
                context_opts=context_opts,
                partition_column=partition_column,
                global_threshold_count=global_threshold_count,
            )
        )

    def to_content(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.ToContent())

    def tqdm(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.Tqdm())

    def passthrough(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.Passthrough())

    def to_document_content_tuple(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.ToDocumentContentTuple())

    def project(self: pipelines.CorpusPipeline, project: Callable[[Any], Any]) -> pipelines.CorpusPipeline:
        return self.add(tasks.Project(project=project))

    def vocabulary(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.Vocabulary())

    def tagged_frame_to_tokens(
        self: pipelines.CorpusPipeline,
        extract_opts: ExtractTaggedTokensOpts,
        filter_opts: TaggedTokensFilterOpts,
    ) -> pipelines.CorpusPipeline:
        return self.add(
            tasks.TaggedFrameToTokens(
                extract_opts=extract_opts,
                filter_opts=filter_opts,
            )
        )
