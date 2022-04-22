from __future__ import annotations

from typing import TYPE_CHECKING

from penelope import corpus as pc

from . import tasks

if TYPE_CHECKING:
    from .. import pipelines

# pylint: disable=no-member


class PipelineShortcutMixIn:
    """Shortcuts for specific tasks that can be injected to derived pipelines"""

    def to_dtm(
        self: pipelines.CorpusPipeline,
        vectorize_opts: pc.VectorizeOpts = None,
        tagged_column: str = None,
    ) -> pipelines.CorpusPipeline:
        """(filename, TEXT => DTM)"""
        return self.add(tasks.ToDTM(vectorize_opts=vectorize_opts or pc.VectorizeOpts(), tagged_column=tagged_column))
