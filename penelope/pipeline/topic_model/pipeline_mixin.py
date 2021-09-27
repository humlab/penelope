from __future__ import annotations

from typing import TYPE_CHECKING

from . import tasks

if TYPE_CHECKING:
    from .. import pipelines


class PipelineShortcutMixIn:
    """Shortcuts for specific tasks that can be injected to derived pipelines"""

    def to_topic_model(
        self: pipelines.CorpusPipeline,
        *,
        corpus_filename: str = None,
        corpus_folder: str = None,
        target_folder: str = None,
        target_name: str = None,
        engine: str = "gensim_lda-multicore",
        engine_args: dict = None,
        store_corpus: bool = False,
        store_compressed: bool = True,
    ) -> pipelines.CorpusPipeline:
        """ TOKENS => TOPIC MODEL """
        return self.add(
            tasks.ToTopicModel(
                corpus_filename=corpus_filename,
                corpus_folder=corpus_folder,
                target_folder=target_folder,
                target_name=target_name,
                engine=engine,
                engine_args=engine_args,
                store_corpus=store_corpus,
                store_compressed=store_compressed,
            )
        )