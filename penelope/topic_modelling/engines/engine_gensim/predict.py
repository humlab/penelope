from __future__ import annotations

from typing import Any, Iterable, Union

import gensim.models as models
import penelope.utility as utility
import penelope.vendor.gensim.wrappers as wrappers
from gensim.matutils import Sparse2Corpus
from penelope.corpus.dtm.corpus import VectorizedCorpus

from .wrappers.mallet_topic_model import MalletTopicModel

# pylint: disable=unused-argument


@utility.deprecated
def gensim_lsi_predict(model: models.LsiModel, corpus: Any, scaled=False, chunk_size=512, **kwargs):
    """Predict using Gensim LsiModel. Corpus must be in BoW format i.e. List[List[(token_id, count)]
    BOW => Iterable
    """
    # data_iter = enumerate(model[corpus, minimum_probability]) same as:
    data_iter = enumerate(model[corpus, scaled, chunk_size])
    return data_iter


def gensim_lda_predict(
    model: models.LdaModel | models.LdaMulticore, corpus: Any, minimum_probability: float = 0.0
) -> Iterable:
    """Predict using Gensim LdaModel. Corpus must be in BoW format i.e. List[List[(token_id, count)]
    BOW => Iterable
    """
    # data_iter = enumerate(model[corpus, minimum_probability]) same as:
    data_iter = enumerate(model.get_document_topics(bow=corpus, minimum_probability=minimum_probability))
    return data_iter


def mallet_lda_predict(model: wrappers.LdaMallet, corpus: Any) -> Iterable:
    # data_iter = enumerate(model.load_document_topics())
    data_iter = enumerate(model[corpus])
    return data_iter


SupportedModels = Union[models.LdaModel, models.LdaMulticore, MalletTopicModel, models.LsiModel]


def predict(model: SupportedModels, corpus: Any, minimum_probability: float = 0.0, **kwargs) -> Iterable:

    minimum_probability: float = kwargs.get('minimum_probability', 0.0)

    if not isinstance(
        model,
        (
            models.LdaMulticore,
            models.LdaModel,
            models.LsiModel,
            MalletTopicModel,
            wrappers.LdaMallet,
        ),
    ):
        raise ValueError(f"Gensim model {type(model)} is not supported")

    if isinstance(corpus, VectorizedCorpus):
        corpus = Sparse2Corpus(corpus.data, documents_columns=False)

    if isinstance(model, (models.LdaMulticore, models.LdaModel)):
        data_iter = gensim_lda_predict(model, corpus, minimum_probability=minimum_probability)
    elif isinstance(model, (MalletTopicModel, wrappers.LdaMallet)) or hasattr(model, 'load_document_topics'):
        data_iter = mallet_lda_predict(model, corpus)
    elif hasattr(model, '__getitem__'):
        data_iter = ((document_id, model[corpus[document_id]]) for document_id in range(0, len(corpus)))
    else:
        raise ValueError("unsupported or deprecated model")

    for document_id, topic_weights in data_iter:
        for (topic_id, weight) in (
            (topic_id, weight) for (topic_id, weight) in topic_weights if weight >= minimum_probability
        ):
            yield (document_id, topic_id, weight)
