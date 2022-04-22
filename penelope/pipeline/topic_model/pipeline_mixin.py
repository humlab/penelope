from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from . import tasks

if TYPE_CHECKING:
    from .. import pipelines

# pylint: disable=no-member


class PipelineShortcutMixIn:
    """Shortcuts for specific tasks that can be injected to derived pipelines"""

    def to_topic_model(
        self: pipelines.CorpusPipeline,
        *,
        train_corpus_folder: str = None,
        trained_model_folder: str = None,
        target_mode: Literal['train', 'predict', 'both'] = 'both',
        target_folder: str = None,
        target_name: str = None,
        engine: str = "not-specified",
        engine_args: dict = None,
        n_tokens: int = 200,
        minimum_probability: float = 0.01,
        store_corpus: bool = False,
        store_compressed: bool = True,
    ) -> pipelines.CorpusPipeline:
        """[summary]

        Args:
            self (pipelines.CorpusPipeline): [description]
            target_mode (str, optional): What to do: train, predict or both. Defaults to None.
            train_corpus_folder (str, optional): Use this prepared trained corpus. Defaults to None.
            trained_model_folder (str, optional): If `predict` then use this existing model. Defaults to None.
            target_folder (str, optional): Where to put the result. Defaults to None.
            target_name (str, optional): Result key/identifier. Defaults to None.
            engine (str, optional): TM engine. Defaults to "not-specified".
            engine_args (dict, optional): TM engine arguments. Defaults to None.
            n_tokens (int, optional): Number of tokens per topic to extract. Defaults to 200.
            minimum_probability (float, optional): Discard topic weights less than value. Defaults to 0.01.
            store_corpus (bool, optional): Store train corpus. Defaults to False.
            store_compressed (bool, optional): Store train corpus compressed. Defaults to True.

        """

        return self.add(
            tasks.ToTopicModel(
                train_corpus_folder=train_corpus_folder,
                trained_model_folder=trained_model_folder,
                target_mode=target_mode,
                target_folder=target_folder,
                target_name=target_name,
                engine=engine,
                engine_args=engine_args,
                n_tokens=n_tokens,
                minimum_probability=minimum_probability,
                store_corpus=store_corpus,
                store_compressed=store_compressed,
            )
        )

    def predict_topics(
        self: pipelines.CorpusPipeline,
        *,
        model_folder: str = None,
        target_folder: str = None,
        target_name: str = None,
        n_tokens: int = 200,
        minimum_probability: float = 0.001,
    ) -> pipelines.CorpusPipeline:
        """TOKENS => TOPIC MODEL"""
        return self.to_topic_model(
            train_corpus_folder=None,
            trained_model_folder=model_folder,
            target_mode='predict',
            target_folder=target_folder,
            target_name=target_name,
            engine="not-specified",
            engine_args=None,
            n_tokens=n_tokens,
            minimum_probability=minimum_probability,
        )
