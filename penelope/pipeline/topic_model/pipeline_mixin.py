from __future__ import annotations

from typing import TYPE_CHECKING

from . import tasks

if TYPE_CHECKING:
    from .. import pipelines

# pylint: disable=no-member


class PipelineShortcutMixIn:
    """Shortcuts for specific tasks that can be injected to derived pipelines"""

    def to_topic_model(
        self: pipelines.CorpusPipeline,
        *,
        corpus_source: str = None,
        train_corpus_folder: str = None,
        target_folder: str = None,
        target_name: str = None,
        engine: str = "not-specified",
        engine_args: dict = None,
        store_corpus: bool = False,
        store_compressed: bool = True,
    ) -> pipelines.CorpusPipeline:
        """ TOKENS => TOPIC MODEL """
        return self.add(
            tasks.ToTopicModel(
                corpus_source=corpus_source,
                train_corpus_folder=train_corpus_folder,
                target_folder=target_folder,
                target_name=target_name,
                engine=engine,
                engine_args=engine_args,
                store_corpus=store_corpus,
                store_compressed=store_compressed,
            )
        )

    def predict_topics(
        self: pipelines.CorpusPipeline,
        *,
        model_folder: str = None,
        model_name: str = None,
        target_folder: str = None,
        target_name: str = None,
    ) -> pipelines.CorpusPipeline:
        """ TOKENS => TOPIC MODEL """
        return self.add(
            tasks.PredictTopics(
                model_folder=model_folder,
                model_name=model_name,
                target_folder=target_folder,
                target_name=target_name,
            )
        )
