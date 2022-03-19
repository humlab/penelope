# type: ignore

from typing import Any, Iterable, Sequence, Type, get_args

from ... import interfaces
from ..interface import ITopicModelEngine
from . import predict, train
from .predict import SupportedModels


def is_supported(model: Any) -> bool:
    return type(model) in get_args(SupportedModels)


class TopicModelEngine(ITopicModelEngine):
    def __init__(self, model: SupportedModels):
        super().__init__(model)

    @staticmethod
    def is_supported(model: Any):
        return is_supported(model)

    @staticmethod
    def supported_models() -> Sequence[Type]:
        return get_args(SupportedModels)

    def n_topics(self) -> int:
        return self.model.n_topics

    def get_topic_token_weights_data(self, n_tokens: int = 200, id2term: dict = None, **_) -> list[tuple[str, float]]:
        if not hasattr(self.model, 'top_topic_terms'):
            raise ValueError(f"{type(self.model)} has no top_topic_terms attribute")

        data = self.model.top_topic_terms(id2term, topics=-1, top_n=n_tokens, weights=True)
        return data

    def top_topic_tokens(
        self, topic_id: int, n_tokens: int = 200, id2term: dict = None, **_
    ) -> list[tuple[str, float]]:
        """Return `n_tokens` top tokens from topic `topic_id`"""

        if not is_supported(self.model):
            raise ValueError(f"{type(self.model)} is not supported")

        topic_words = list(self.model.top_topic_terms(id2term, topics=(topic_id,), top_n=n_tokens, weights=True))

        if len(topic_words) == 0:
            return []

        return topic_words[0][1]

    @staticmethod
    def train(
        train_corpus: interfaces.TrainingCorpus,
        method: str,
        engine_args: dict[str, Any],
        **kwargs: dict[str, Any],
    ) -> interfaces.InferredModel:
        return train.train(train_corpus=train_corpus, method=method, engine_args=engine_args, **kwargs)

    def predict(self, corpus: Any, minimum_probability: float = 0.0, **kwargs) -> Iterable:
        predict.predict(self.model, corpus=corpus, minimum_probability=minimum_probability, **kwargs)
