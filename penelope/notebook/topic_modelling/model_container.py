import penelope.topic_modelling as topic_modelling


class TopicModelException(Exception):
    pass


class TopicModelContainer:
    """Class for current (last) computed or loaded model"""

    _singleton = None

    def __init__(
        self,
        _inferred_data: topic_modelling.InferredModel = None,
        _inferred_topics: topic_modelling.InferredTopicsData = None,
    ):
        self._inferred_data: topic_modelling.InferredModel = None
        self._inferred_topics: topic_modelling.InferredTopicsData = None

    @staticmethod
    def singleton():
        TopicModelContainer._singleton = TopicModelContainer._singleton or TopicModelContainer()
        return TopicModelContainer._singleton

    def set_data(
        self, _inferred_data: topic_modelling.InferredModel, _inferred_topics: topic_modelling.InferredTopicsData
    ):
        """ Fix missing document attribute n_terms """
        if 'n_terms' not in _inferred_topics.document_index.columns:
            assert _inferred_data.train_corpus is not None
            _inferred_topics.document_index['n_terms'] = _inferred_data.train_corpus.corpus.sparse.sum(axis=0).A1

        self._inferred_data = _inferred_data
        self._inferred_topics = _inferred_topics

    @property
    def inferred_model(self) -> topic_modelling.InferredModel:
        if self._inferred_data is None:
            raise TopicModelException('Model not loaded or computed')
        return self._inferred_data

    @property
    def inferred_topics(self) -> topic_modelling.InferredTopicsData:
        return self._inferred_topics

    @property
    def topic_model(self):
        return self.inferred_model.topic_model

    @property
    def id2term(self):
        # return self.inferred_model.train_corpus.id2word
        return self.inferred_topics.id2word

    @property
    def num_topics(self) -> int:
        return int(self.inferred_topics.topic_token_overview.index.max()) + 1
        # model = self.inferred_model.topic_model
        # if model is None:
        #     return 0
        # if hasattr(model, 'num_topics'):
        #     return model.num_topics
        # if hasattr(model, 'n_topics'):
        #     return model.n_topics

        # return 0

    # @property
    # def relevant_topics(self):
    #     return self._inferred_topics.relevant_topic_ids
