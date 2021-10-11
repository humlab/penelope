# type: ignore

from typing import Any, Dict, Iterable, List, Sequence, Tuple, Type, get_args

from .. import interfaces
from . import predict, train
from .options import SUPPORTED_ENGINES
from .predict import SupportedModels
from .utility import malletmodel2ldamodel


def is_supported(model: Any) -> bool:
    return type(model) in get_args(SupportedModels)


class TopicModelEngine(interfaces.ITopicModelEngine):
    def __init__(self, model: SupportedModels):
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

    def topics_tokens(self, n_tokens: int = 200, id2term: dict = None, **_) -> List[Tuple[float, str]]:

        if not is_supported(self.model):
            raise ValueError(f"{type(self.model)} is not supported")

        if not hasattr(self.model, 'show_topics'):
            raise ValueError(f"{type(self.model)} has no show_topics attribute")

        return self.model.show_topics(num_topics=-1, num_words=n_tokens, formatted=False)

    def topic_tokens(self, topic_id: int, n_tokens: int = 200, id2term: dict = None, **_) -> List[Tuple[str, float]]:
        """Return `n_tokens` top tokens from topic `topic_id`"""

        if not is_supported(self.model):
            raise ValueError(f"{type(self.model)} is not supported")

        return self.model.show_topic(topic_id, topn=n_tokens)

    @staticmethod
    def train(
        train_corpus: interfaces.TrainingCorpus, method: str, engine_args: Dict[str, Any], **kwargs: Dict[str, Any]
    ) -> interfaces.InferredModel:
        return train.train(train_corpus=train_corpus, method=method, engine_args=engine_args, **kwargs)

    def predict(self, corpus: Any, minimum_probability: float = 0.0, **kwargs) -> Iterable:
        return predict.predict(self.model, corpus=corpus, minimum_probability=minimum_probability, **kwargs)
