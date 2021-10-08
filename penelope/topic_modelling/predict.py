from __future__ import annotations

from typing import Any

import gensim.corpora as corpora
import numpy as np
import pandas as pd
from gensim.matutils import Sparse2Corpus
from penelope.corpus import DocumentIndex, DocumentIndexHelper, Token2Id, VectorizedCorpus

from .interfaces import DocumentTopicsWeightsIter, InferredTopicsData, ITopicModelEngine
from .utility import add_document_terms_count, get_engine_by_model_type

# pylint: disable=unused-argument


def document_topics_iter_to_dataframe(document_index: pd.DataFrame, data: DocumentTopicsWeightsIter) -> pd.DataFrame:
    """Convert document-topic-weight stream into a data frame."""
    document_topics = pd.DataFrame(data, columns=['document_id', 'topic_id', 'weight'])
    document_topics['document_id'] = document_topics.document_id.astype(np.uint32)
    document_topics['topic_id'] = document_topics.topic_id.astype(np.uint16)
    document_topics = DocumentIndexHelper(document_index).overload(document_topics, 'year')
    return document_topics


def predict_document_topics(
    model: Any,
    corpus: Any,
    document_index: DocumentIndex = None,
    minimum_probability: float = 0.001,
    **kwargs,
) -> pd.DataFrame:
    """Predict topocs fpr `corpus`using `model`. Return document-topic dataframe.

    Args:
        model (Any): The topic model.
        corpus (Any): The corpus to predict topics on.
        document_index (DocumentIndex, optional): [description]. Defaults to None.
        minimum_probability (float, optional): [description]. Defaults to 0.001.

    Returns:
        pd.DataFrame:  Document topics
    """

    document_topic_weights: pd.DataFrame = document_topics_iter_to_dataframe(
        document_index,
        get_engine_by_model_type(model).predict(corpus, minimum_probability, **kwargs),
    )

    return document_topic_weights


def predict_topics(
    topic_model: Any,
    *,
    corpus: Sparse2Corpus | VectorizedCorpus,
    id2token: corpora.Dictionary | dict | Token2Id,
    document_index: DocumentIndex,
    n_tokens: int = 200,
    minimum_probability: float = 0.001,
    **kwargs,
) -> InferredTopicsData:
    """Predict topics for `corpus`. Return InferredTopicsData.

    Args:
        topic_model (Any): [description]
        corpus (Sparse2Corpus): Corpus to be predicted.
        id2token (corpora.Dictionary): id-to-token mapping
        document_index (DocumentIndex): [description]
        n_tokens (int, optional): [description]. Defaults to 200.
        minimum_probability (float, optional): [description]. Defaults to 0.001.
    Kwargs:
        topic_token_weights (pd.DataFrame, optional): existing topic token distrubution. Defaults to None.
        topic_token_overview (pd.DataFrame, optional): existing overview. Defaults to None.
    """

    if not isinstance(
        corpus,
        (
            Sparse2Corpus,
            VectorizedCorpus,
        ),
    ):
        raise ValueError(f"expected `Sparse2Corpus` or `VectorizedCorpus`, got `{type(corpus)}`")

    if isinstance(corpus, VectorizedCorpus):
        corpus: Sparse2Corpus = Sparse2Corpus(corpus.data)

    engine: ITopicModelEngine = get_engine_by_model_type(topic_model)

    topic_token_weights: pd.DataFrame = (
        kwargs.get('topic_token_weights')
        if kwargs.get('topic_token_weights') is not None
        else engine.get_topic_token_weights(vocabulary=id2token, n_tokens=n_tokens)
    )

    topic_token_overview: pd.DataFrame = (
        kwargs.get('topic_token_overview')
        if kwargs.get('topic_token_overview')
        else engine.get_topic_token_overview(topic_token_weights, n_tokens=n_tokens)
    )

    document_index: pd.DataFrame = add_document_terms_count(document_index, corpus)

    document_topic_weights: pd.DataFrame = predict_document_topics(
        topic_model, corpus, document_index=document_index, minimum_probability=minimum_probability
    )

    topics_data: InferredTopicsData = InferredTopicsData(
        document_index,
        Token2Id.id2token_to_dataframe(id2token),
        topic_token_weights,
        topic_token_overview,
        document_topic_weights,
    )
    return topics_data
