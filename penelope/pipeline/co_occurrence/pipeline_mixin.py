from __future__ import annotations

from typing import TYPE_CHECKING

from . import tasks

if TYPE_CHECKING:
    from penelope.co_occurrence import ContextOpts

    from .. import pipelines


class PipelineShortcutMixIn:
    """Shortcuts for specific tasks that can be injected to derived pipelines"""

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
