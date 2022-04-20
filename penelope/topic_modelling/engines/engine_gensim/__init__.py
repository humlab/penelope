# type: ignore

from typing import Any, Iterable, Sequence, Type, get_args

import pandas as pd

from ... import interfaces
from ..interface import ITopicModelEngine
from . import predict, train

try:
    from .options import SUPPORTED_ENGINES
except:  # pylint: disable=bare-except
    SUPPORTED_ENGINES = {}

try:
    from .predict import SupportedModels
except:  # pylint: disable=bare-except
    SupportedModels = object

try:
    from .utility import diagnostics_to_topic_token_weights_data, malletmodel2ldamodel
except:  # pylint: disable=bare-except

    def malletmodel2ldamodel(
        mallet_model: Any, gamma_threshold: float = 0.001, iterations: int = 50  # pylint: disable=unused-argument
    ):
        return None


def is_supported(model: Any) -> bool:
    return type(model) in get_args(SupportedModels)


class TopicModelEngine(ITopicModelEngine):
    def __init__(self, model: SupportedModels):  # pylint: disable=useless-super-delegation
        super().__init__(model)

    @staticmethod
    def is_supported(model: Any):
        return is_supported(model)

    @staticmethod
    def supported_models() -> Sequence[Type]:
        return get_args(SupportedModels)

    def n_topics(self) -> int:

        if hasattr(self.model, 'num_topics'):
            return self.model.num_topics

        if hasattr(self.model, 'm_T'):
            return self.model.m_T

        raise ValueError(f"{type(self.model)} is not supported")

    def get_topic_token_weights_data(
        self, n_tokens: int = 200, id2term: dict[int, str] = None, **kwargs  # pylint: disable=unused-argument
    ) -> list[tuple[int, tuple[str, float]]]:

        if not is_supported(self.model):
            raise ValueError(f"{type(self.model)} is not supported")

        if kwargs.get('use_diagnostics', True) and self.topic_token_diagnostics is not None:
            return diagnostics_to_topic_token_weights_data(self.topic_token_diagnostics, n_tokens)

        if not hasattr(self.model, 'show_topics'):
            raise ValueError(f"{type(self.model)} has no show_topics attribute")

        return self.model.show_topics(num_topics=-1, num_words=n_tokens, formatted=False)

    def top_topic_tokens(
        self, topic_id: int, n_tokens: int = 200, id2term: dict = None, **_  # pylint: disable=unused-argument
    ) -> list[tuple[str, float]]:
        """Return `n_tokens` top tokens from topic `topic_id`"""

        if not is_supported(self.model):
            raise ValueError(f"{type(self.model)} is not supported")

        return self.model.show_topic(topic_id, topn=n_tokens)

    @staticmethod
    def train(
        train_corpus: interfaces.TrainingCorpus, method: str, engine_args: dict[str, Any], **kwargs: dict[str, Any]
    ) -> interfaces.InferredModel:
        return train.train(train_corpus=train_corpus, method=method, engine_args=engine_args, **kwargs)

    def predict(self, corpus: Any, minimum_probability: float = 0.0, **kwargs) -> Iterable:
        return predict.predict(self.model, corpus=corpus, minimum_probability=minimum_probability, **kwargs)
