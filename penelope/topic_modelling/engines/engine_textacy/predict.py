from __future__ import annotations

from typing import Iterable

import scipy.sparse as sp

from penelope.corpus.dtm.corpus import VectorizedCorpus
from penelope.utility import deprecated
from penelope.vendor import textacy_api

# pylint: disable=unused-argument


@deprecated
def scikit_predict(model: textacy_api.TopicModel, dtm: sp.spmatrix, top_n: int = 10009):
    """scikit-learn, Corpus must be a DTM (e.g. Gensim sparse corpus).  Returns a matrice.

    BOW => Natrice

    """
    if not hasattr(model, 'top_doc_topics'):
        raise ValueError("top_doc_topics")
    data_iter = model.transform(doc_term_matrix=dtm, docs=-1, top_n=top_n, weights=True)
    return data_iter


SupportedModels = None


@deprecated
def predict(
    model: SupportedModels, corpus: VectorizedCorpus, minimum_probability: float = 0.0, top_n: int = 100, **kwargs
) -> Iterable:

    data_iter = scikit_predict(model, corpus.bag_term_matrix, top_n)

    for document_id, topic_weights in data_iter:
        for (topic_id, weight) in (
            (topic_id, weight) for (topic_id, weight) in topic_weights if weight >= minimum_probability
        ):
            yield (document_id, topic_id, weight)
