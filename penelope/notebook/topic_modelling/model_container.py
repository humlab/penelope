from functools import cached_property
from typing import Optional, Union

import penelope.topic_modelling as topic_modelling


class TopicModelException(Exception):
    pass


class TopicModelContainer:
    """Class for current (last) computed or loaded model"""

    _singleton = None

    def __init__(
        self,
        _trained_model: topic_modelling.InferredModel = None,
        _inferred_topics: topic_modelling.InferredTopicsData = None,
        _train_corpus_folder: str = None,
    ):
        self._trained_model: topic_modelling.InferredModel = _trained_model
        self._inferred_topics: topic_modelling.InferredTopicsData = _inferred_topics
        self._train_corpus_folder: str = _train_corpus_folder

    @staticmethod
    def singleton():
        TopicModelContainer._singleton = TopicModelContainer._singleton or TopicModelContainer()
        return TopicModelContainer._singleton

    def set_data(
        self,
        _trained_model: Optional[topic_modelling.InferredModel],
        _inferred_topics: Optional[topic_modelling.InferredTopicsData],
        _train_corpus_folder: Union[str, topic_modelling.TrainingCorpus] = None,
    ):
        if 'n_tokens' not in _inferred_topics.document_index.columns:
            raise ValueError("expected n_tokens in document_index (previous fix is removed)")
            # assert _trained_model.train_corpus is not None
            # _inferred_topics.document_index['n_tokens'] = _trained_model.train_corpus.n_tokens

        self._trained_model = _trained_model
        self._inferred_topics = _inferred_topics
        self._train_corpus_folder = _train_corpus_folder

    @cached_property
    def train_corpus(self) -> topic_modelling.TrainingCorpus:
        if not self._train_corpus_folder:
            raise TopicModelException('Training corpus folder is not set!')
        if isinstance(self._train_corpus_folder, topic_modelling.TrainingCorpus):
            return self._train_corpus_folder
        return topic_modelling.TrainingCorpus.load(self._train_corpus_folder)

    @property
    def trained_model(self) -> topic_modelling.InferredModel:
        if self._trained_model is None:
            raise TopicModelException('Model not loaded or computed')
        return self._trained_model

    @property
    def inferred_topics(self) -> topic_modelling.InferredTopicsData:
        return self._inferred_topics

    @property
    def topic_model(self):
        return self.trained_model.topic_model

    @property
    def id2term(self):
        return self.inferred_topics.id2word

    @property
    def num_topics(self) -> int:
        return self.inferred_topics.num_topics
