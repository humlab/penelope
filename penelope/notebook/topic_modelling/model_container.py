from typing import Any, Callable, Literal, Self

import penelope.topic_modelling as tm


class TopicModelException(Exception):
    pass


class TopicModelContainer:
    """Class for current (last) computed or loaded model"""

    _singleton: "TopicModelContainer" = None

    def __init__(self):
        self._folder: str = None
        self._trained_model: tm.InferredModel = None
        self._inferred_topics: tm.InferredTopicsData = None
        self._train_corpus_folder: str | tm.TrainingCorpus = None
        self._folder: str = None
        self._observers: dict[Any, Callable] = {}
        self._property_bag: dict[str, Any] = {}

    def register(self, observer: Any, callback: Callable) -> None:
        self._observers[observer] = callback

    def notify(self) -> None:
        for observer, callback in self._observers.items():
            callback(self, observer)

    def __getitem__(
        self, key: Literal['topic_model', 'trained_model', 'topics_data', 'inferred_topics']
    ) -> tm.InferredModel | tm.InferredModel:
        if key == 'trained_model':
            return self.trained_model
        if key == 'topic_model':
            return self.trained_model.topic_model
        if key in ['inferred_topics', 'topics_data']:
            return self.inferred_topics
        return self.__dict__[key]

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    @staticmethod
    def singleton() -> "TopicModelContainer":
        TopicModelContainer._singleton = TopicModelContainer._singleton or TopicModelContainer()
        return TopicModelContainer._singleton

    def update(
        self,
        *,
        inferred_topics: tm.InferredTopicsData | None = None,
        folder: str | None = None,
        trained_model: tm.InferredModel | None = None,
        train_corpus_folder: str | tm.TrainingCorpus | None = None,
    ) -> Self:
        self._inferred_topics = inferred_topics
        self._folder = folder or train_corpus_folder
        self._trained_model = trained_model
        self._train_corpus_folder = train_corpus_folder
        self.notify()
        return self

    def load(self, *, folder: str, slim: bool = False, lazy: bool = True) -> None:
        trained_model: tm.InferredModel = tm.InferredModel.load(folder, lazy=lazy)
        inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=folder, slim=slim)

        self.update(
            inferred_topics=inferred_topics,
            folder=folder,
            trained_model=trained_model,
            train_corpus_folder=folder,
        )

    @property
    def trained_model(self) -> tm.InferredModel:
        if self._trained_model is None:
            raise TopicModelException('Model not loaded or computed')
        return self._trained_model

    @property
    def inferred_topics(self) -> tm.InferredTopicsData:
        return self._inferred_topics

    @property
    def topic_model(self) -> Any:
        return self.trained_model.topic_model

    @property
    def train_corpus_folder(self) -> str:
        return self._folder

    @property
    def folder(self) -> str:
        return self._folder

    def get(self, key: str, default: Any = None) -> Any:
        return self._property_bag.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._property_bag[key] = value
