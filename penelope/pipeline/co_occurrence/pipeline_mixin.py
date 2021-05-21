from __future__ import annotations

from typing import TYPE_CHECKING

from . import tasks

if TYPE_CHECKING:
    from penelope.co_occurrence import ContextOpts

    from .. import pipelines


class PipelineShortcutMixIn:
    """Shortcuts for specific tasks that can be injected to derived pipelines"""

    # @utility.deprecated
    # def to_corpus_co_occurrence(
    #     self: pipelines.CorpusPipeline,
    #     *,
    #     context_opts: ContextOpts = None,
    #     transform_opts: TokensTransformOpts = None,
    #     global_threshold_count: int = None,
    # ) -> pipelines.CorpusPipeline:
    #     """ (filename, DOCUMENT_CONTENT_TUPLES => DATAFRAME) """
    #     return self.add(
    #         tasks.ToCorpusCoOccurrence(
    #             context_opts=context_opts,
    #             transform_opts=transform_opts,
    #             global_threshold_count=global_threshold_count,
    #         )
    #     )

    def to_document_co_occurrence(
        self: pipelines.CorpusPipeline,
        *,
        context_opts: ContextOpts = None,
        ingest_tokens: bool = True,
    ) -> pipelines.CorpusPipeline:
        """ TOKENS => CO_OCCURRENCE_DTM_DOCUMENT """
        return self.add(
            tasks.ToCoOccurrenceDTM(
                context_opts=context_opts,
                ingest_tokens=ingest_tokens,
            )
        )

    def to_corpus_co_occurrence(
        self: pipelines.CorpusPipeline,
        *,
        context_opts: ContextOpts = None,
        global_threshold_count: int = 1,
    ) -> pipelines.CorpusPipeline:
        """ TOKENS => CO_OCCURRENCE_DTM_CORPUS """
        return self.add(
            tasks.ToCorpusCoOccurrenceDTM(
                context_opts=context_opts,
                global_threshold_count=global_threshold_count,
            )
        )
