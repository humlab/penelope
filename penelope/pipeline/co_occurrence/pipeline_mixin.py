from __future__ import annotations

from typing import TYPE_CHECKING

from . import tasks

if TYPE_CHECKING:
    from penelope.co_occurrence import ContextOpts

    from .. import pipelines

# pylint: disable=no-member


class PipelineShortcutMixIn:
    """Shortcuts for specific tasks that can be injected to derived pipelines"""

    def to_document_co_occurrence(
        self: pipelines.CorpusPipeline, *, context_opts: ContextOpts = None
    ) -> pipelines.CorpusPipeline:
        """TOKENS => CO_OCCURRENCE_DTM_DOCUMENT"""
        return self.add(tasks.ToCoOccurrenceDTM(context_opts=context_opts))

    def to_corpus_co_occurrence(
        self: pipelines.CorpusPipeline,
        *,
        context_opts: ContextOpts = None,
    ) -> pipelines.CorpusPipeline:
        """TOKENS => CO_OCCURRENCE_DTM_CORPUS"""
        return self.add(tasks.ToCorpusCoOccurrenceDTM(context_opts=context_opts))
