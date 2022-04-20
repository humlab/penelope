from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from penelope import corpus as pc
from penelope.corpus import dtm
from penelope.vendor.gensim_api import corpora as gensim_corpora

from .engines import get_engine_by_model_type
from .topics_data import InferredTopicsData

if TYPE_CHECKING:
    from penelope.type_alias import DocumentTopicsWeightsIter

    from .engines.interface import ITopicModelEngine


# pylint: disable=unused-argument


def to_dataframe(document_index: pd.DataFrame, data: DocumentTopicsWeightsIter) -> pd.DataFrame:
    """Convert document-topic-weight stream into a data frame."""
    # REDUCE MEMORY CONSUMPTION
    # MAKE WEIGHT float32 (or float 16)
    # START WITH AN EMPTY FRAME!!!!

    document_topics = pd.DataFrame(data, columns=['document_id', 'topic_id', 'weight'])
    document_topics['document_id'] = document_topics.document_id.astype(np.uint32)
    document_topics['topic_id'] = document_topics.topic_id.astype(np.uint16)
    document_topics = pc.DocumentIndexHelper(document_index).overload(document_topics, 'year')
    return document_topics


def predict_topics(
    topic_model: Any,
    *,
    corpus: gensim_corpora.Sparse2Corpus | pc.VectorizedCorpus,
    id2token: Mapping[int, str] | dict | pc.Token2Id,
    document_index: pc.DocumentIndex = None,
    n_tokens: int = 200,
    minimum_probability: float = 0.001,
    **kwargs,
) -> InferredTopicsData:
    """Predict topics for `corpus`. Return InferredTopicsData.

    Args:
        topic_model (Any): [description]
        corpus (Sparse2Corpus): Corpus to be predicted.
        id2token (corpora.Dictionary): id-to-token mapping
        document_index (DocumentIndex): Document index
        n_tokens (int, optional): Number of tokens per topic to keep. Defaults to 200.
        minimum_probability (float, optional): Minimum doc-topic weights to keep. Defaults to 0.001.
    Kwargs:
        topic_token_weights (pd.DataFrame, optional): existing topic token distrubution. Defaults to None.
        topic_token_overview (pd.DataFrame, optional): existing overview. Defaults to None.
    """

    vectorized_corpus: pc.VectorizedCorpus = dtm.TranslateCorpus.translate(
        corpus,
        token2id=pc.id2token2token2id(id2token),
        document_index=document_index,
        vectorize_opts=pc.VectorizeOpts().update(**kwargs),
    )

    engine: ITopicModelEngine = get_engine_by_model_type(topic_model)

    document_topic_weights: DocumentTopicsWeightsIter = engine.predict(
        vectorized_corpus, minimum_probability=minimum_probability, **kwargs
    )

    topic_token_weights: pd.DataFrame = (
        kwargs.get('topic_token_weights')
        if kwargs.get('topic_token_weights') is not None
        else engine.get_topic_token_weights(vocabulary=vectorized_corpus.id2token, n_tokens=n_tokens)
    )

    topic_token_overview: pd.DataFrame = (
        kwargs.get('topic_token_overview')
        if kwargs.get('topic_token_overview') is not None
        else engine.get_topic_token_overview(topic_token_weights, n_tokens=n_tokens)
    )

    topic_diagnostics: pd.DataFrame = kwargs.get('topic_diagnostics', engine.topic_diagnostics)
    topic_token_diagnostics: pd.DataFrame = kwargs.get('topic_token_diagnostics', engine.topic_token_diagnostics)

    document_index: pd.DataFrame = pc.update_document_index_token_counts_by_corpus(document_index, vectorized_corpus)

    topics_data: InferredTopicsData = InferredTopicsData(
        dictionary=pc.Token2Id.id2token_to_dataframe(vectorized_corpus.id2token),
        topic_token_weights=topic_token_weights,
        topic_token_overview=topic_token_overview,
        document_index=document_index,
        document_topic_weights=to_dataframe(document_index, document_topic_weights),
        topic_diagnostics=topic_diagnostics,
        token_diagnostics=topic_token_diagnostics,
    )
    return topics_data
