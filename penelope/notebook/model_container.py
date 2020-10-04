class TopicModelException(Exception):
    pass


class TopicModelContainer:
    """Class for current (last) computed or loaded model"""

    _singleton = None

    def __init__(self):
        self._model_data = None
        self._compiled_data = None

    @staticmethod
    def singleton():
        TopicModelContainer._singleton = TopicModelContainer._singleton or TopicModelContainer()
        return TopicModelContainer._singleton

    def set_data(self, m_data, c_data):
        """ Fix missing document attribute n_terms """
        if 'n_terms' not in c_data.documents.columns:
            c_data.documents['n_terms'] = m_data.corpus.sparse.sum(axis=0).A1

        self._model_data = m_data
        self._compiled_data = c_data

    @property
    def model_data(self):
        if self._model_data is None:
            raise TopicModelException('Model not loaded or computed')
        return self._model_data

    @property
    def compiled_data(self):
        return self._compiled_data

    @property
    def topic_model(self):
        return self.model_data.topic_model

    @property
    def id2term(self):
        return self.model_data.id2term

    @property
    def num_topics(self):
        tm = self.topic_model
        if tm is None:
            return 0
        if hasattr(tm, 'num_topics'):
            return tm.num_topics
        if hasattr(tm, 'n_topics'):
            return tm.n_topics

        return 0

    @property
    def relevant_topics(self):
        return self._compiled_data.relevant_topic_ids
