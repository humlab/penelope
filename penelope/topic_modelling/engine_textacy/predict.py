from __future__ import annotations

from typing import Any, Iterable

import textacy.tm as tm
from gensim.matutils import Sparse2Corpus
from penelope.utility import deprecated

# pylint: disable=unused-argument


@deprecated
def scikit_predict(model: tm.TopicModel, corpus: Sparse2Corpus, top_n: int = 10009):
    """scikit-learn, Corpus must be a DTM (e.g. Gensim sparse corpus).  Returns a matrice.

    BOW => Natrice

    """
    if not hasattr(model, 'top_doc_topics'):
        raise ValueError("top_doc_topics")
    if not isinstance(corpus, Sparse2Corpus):
        raise ValueError("Only Sparse2Corpus valid for inference!")
    data_iter = model.transform(doc_term_matrix=corpus.sparse, docs=-1, top_n=top_n, weights=True)
    # print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return data_iter


SupportedModels = None


@deprecated
def predict(
    model: SupportedModels, corpus: Any, minimum_probability: float = 0.0, top_n: int = 100, **kwargs
) -> Iterable:

    data_iter = scikit_predict(model, corpus, top_n)

    for document_id, topic_weights in data_iter:
        for (topic_id, weight) in (
            (topic_id, weight) for (topic_id, weight) in topic_weights if weight >= minimum_probability
        ):
            yield (document_id, topic_id, weight)
