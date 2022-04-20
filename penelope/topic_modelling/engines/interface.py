from __future__ import annotations

import abc
from functools import cached_property
from typing import TYPE_CHECKING, Any, Iterable, Sequence, Type

import numpy as np
import pandas as pd

from penelope.corpus import Token2Id
from penelope.type_alias import TokenWeights

if TYPE_CHECKING:
    from ..interfaces import InferredModel, TrainingCorpus


class EngineSpec:
    def __init__(self, key: str, engine: Type[Any]):
        self.key: str = key
        self.engine: Type[Any] = engine

    def get_options(
        self,
        corpus: Any,  # pylint: disable=unused-argument
        id2word: dict[int, str],  # pylint: disable=unused-argument
        engine_args: dict,  # pylint: disable=unused-argument
    ) -> dict:
        return dict()

    @property
    def algorithm(self):
        return self.key.split('_', maxsplit=1)[1].upper()

    @property
    def engine_name(self):
        return self.key.split('_', maxsplit=1)[0].title()

    @property
    def description(self):
        return f"{self.engine_name} {self.algorithm.replace('-', ' ')}"


class ITopicModelEngine(abc.ABC):
    def __init__(self, model: Any):
        self.model = model

    @staticmethod
    @abc.abstractmethod
    def is_supported(model: Any) -> bool:
        ...

    @staticmethod
    @abc.abstractmethod
    def supported_models() -> Sequence[Type]:
        ...

    @abc.abstractmethod
    def n_topics(self) -> int:
        ...

    @abc.abstractmethod
    def get_topic_token_weights_data(
        self, n_tokens: int = 200, id2term: dict = None, **_
    ) -> list[tuple[int, list[tuple[str, float]]]]:
        ...

    @abc.abstractmethod
    def top_topic_tokens(
        self, topic_id: int, n_tokens: int = 200, id2term: dict = None, **_
    ) -> list[tuple[int, TokenWeights]]:
        ...

    @staticmethod
    @abc.abstractmethod
    def train(
        train_corpus: "TrainingCorpus", method: str, engine_args: dict[str, Any], **kwargs: dict[str, Any]
    ) -> "InferredModel":
        ...

    @abc.abstractmethod
    def predict(self, corpus: Any, minimum_probability: float = 0.005, **kwargs) -> Iterable:
        ...

    def get_topic_token_weights(
        self, vocabulary: Any, n_tokens: int = 200, minimum_probability: float = 0.000001
    ) -> pd.DataFrame:
        """Compile document topic weights. Return DataFrame."""
        id2token: dict = Token2Id.any_to_id2token(vocabulary)
        topic_data: list[tuple[int, TokenWeights]] = self.get_topic_token_weights_data(
            n_tokens=n_tokens, id2term=id2token
        )

        topic_token_weights: pd.DataFrame = pd.DataFrame(
            [
                (topic_id, token, weight)
                for topic_id, tokens in topic_data
                for token, weight in tokens
                if weight > minimum_probability
            ],
            columns=['topic_id', 'token', 'weight'],
        )

        topic_token_weights['topic_id'] = topic_token_weights.topic_id.astype(np.uint16)

        fg = {v: k for k, v in id2token.items()}.get

        topic_token_weights['token_id'] = topic_token_weights.token.apply(fg)

        return topic_token_weights[['topic_id', 'token_id', 'token', 'weight']]

    def get_topic_token_overview(self, topic_token_weights: pd.DataFrame, n_tokens: int = 200) -> pd.DataFrame:
        """
        Group by topic_id and concatenate n_tokens words within group sorted by weight descending.
        There must be a better way of doing this...
        """

        alpha: list[float] = self.model.alpha if 'alpha' in self.model.__dict__ else None

        overview: pd.DataFrame = (
            topic_token_weights.groupby('topic_id')
            .apply(lambda x: sorted(list(zip(x["token"], x["weight"])), key=lambda z: z[1], reverse=True))
            .apply(lambda x: ' '.join([z[0] for z in x][:n_tokens]))
            .reset_index()
        )
        overview.columns = ['topic_id', 'tokens']
        overview['alpha'] = overview.topic_id.apply(lambda topic_id: alpha[topic_id]) if alpha is not None else 0.0
        overview = overview.set_index('topic_id')

        if self.topic_diagnostics is not None:
            overview = overview.merge(self.topic_diagnostics, left_index=True, right_index=True, how='left')

        return overview

    @cached_property
    def topic_diagnostics(self) -> pd.DataFrame:
        if hasattr(self.model, 'load_topic_diagnostics'):
            return self.model.load_topic_diagnostics()
        return None

    @cached_property
    def topic_token_diagnostics(self) -> pd.DataFrame:
        if hasattr(self.model, 'load_topic_token_diagnostics'):
            return self.model.load_topic_token_diagnostics()
        return None
