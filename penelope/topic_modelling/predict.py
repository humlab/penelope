from __future__ import annotations

from typing import Any

import gensim.corpora as corpora
import numpy as np
import pandas as pd
from gensim.matutils import Sparse2Corpus
from loguru import logger
from penelope.corpus import DocumentIndex, DocumentIndexHelper, Token2Id, VectorizedCorpus

from .interfaces import DocumentTopicsWeightsIter, InferredTopicsData, ITopicModelEngine
from .utility import get_engine_by_model_type

# pylint: disable=unused-argument


def to_dataframe(document_index: pd.DataFrame, data: DocumentTopicsWeightsIter) -> pd.DataFrame:
    """Convert document-topic-weight stream into a data frame."""
    document_topics = pd.DataFrame(data, columns=['document_id', 'topic_id', 'weight'])
    document_topics['document_id'] = document_topics.document_id.astype(np.uint32)
    document_topics['topic_id'] = document_topics.topic_id.astype(np.uint16)
    document_topics = DocumentIndexHelper(document_index).overload(document_topics, 'year')
    return document_topics


def predict_topics(
    topic_model: Any,
    *,
    corpus: Sparse2Corpus | VectorizedCorpus,
    id2token: corpora.Dictionary | dict | Token2Id,
    document_index: DocumentIndex = None,
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

    if not isinstance(corpus, (Sparse2Corpus, VectorizedCorpus)):
        raise ValueError(f"expected `Sparse2Corpus` or `VectorizedCorpus`, got `{type(corpus)}`")

    if isinstance(corpus, VectorizedCorpus):

        if document_index is not None:
            if document_index is not corpus.document_index:
                logger.warning("using corpus document index (ignoring supplied document index)")

        id2token: dict = id2token or corpus.id2token
        document_index: dict = corpus.document_index
        corpus: Sparse2Corpus = Sparse2Corpus(corpus.data, documents_columns=False)

    if isinstance(id2token, (corpora.Dictionary, Token2Id)):
        """We only need the dict"""
        id2token = id2token.id2token

    engine: ITopicModelEngine = get_engine_by_model_type(topic_model)

    topic_token_weights: pd.DataFrame = (
        kwargs.get('topic_token_weights')
        if kwargs.get('topic_token_weights') is not None
        else engine.get_topic_token_weights(vocabulary=id2token, n_tokens=n_tokens)
    )

    topic_token_overview: pd.DataFrame = (
        kwargs.get('topic_token_overview')
        if kwargs.get('topic_token_overview') is not None
        else engine.get_topic_token_overview(topic_token_weights, n_tokens=n_tokens)
    )

    document_index: pd.DataFrame = DocumentIndexHelper(document_index).update_counts_by_corpus(corpus).document_index

    data: DocumentTopicsWeightsIter = engine.predict(corpus, minimum_probability, **kwargs)

    topics_data: InferredTopicsData = InferredTopicsData(
        dictionary=Token2Id.id2token_to_dataframe(id2token),
        topic_token_weights=topic_token_weights,
        topic_token_overview=topic_token_overview,
        document_index=document_index,
        document_topic_weights=to_dataframe(document_index, data),
    )
    return topics_data
