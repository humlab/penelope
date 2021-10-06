from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from gensim.matutils import Sparse2Corpus
from penelope.corpus import DocumentIndex, DocumentIndexHelper, VectorizedCorpus

from .interfaces import DocumentTopicsWeightsIter, InferredTopicsData, ITopicModelEngine
from .utility import add_document_terms_count, get_engine_by_model_type

# pylint: disable=unused-argument


def predict_document_topics(
    model: Any,
    corpus: Any,
    document_index: DocumentIndex = None,
    minimum_probability: float = 0.001,
    **kwargs,
) -> pd.DataFrame:
    """Applies a the topic model on `corpus` and returns a document-topic dataframe

    Args:
        model (Any): The topic model.
        corpus (Any): The corpus to predict topics on.
        document_index (DocumentIndex, optional): [description]. Defaults to None.
        minimum_probability (float, optional): [description]. Defaults to 0.001.

    Returns:
        pd.DataFrame:  Document topics
    """
    engine: ITopicModelEngine = get_engine_by_model_type(model)

    data: DocumentTopicsWeightsIter = engine.predict(corpus, minimum_probability, **kwargs)

    return _to_dataframe(document_index, data)


def _to_dataframe(document_index: pd.DataFrame, data: DocumentTopicsWeightsIter) -> pd.DataFrame:

    document_topics = pd.DataFrame(data, columns=['document_id', 'topic_id', 'weight'])

    document_topics['document_id'] = document_topics.document_id.astype(np.uint32)
    document_topics['topic_id'] = document_topics.topic_id.astype(np.uint16)

    document_topics = DocumentIndexHelper(document_index).overload(document_topics, 'year')

    return document_topics


def _id2word_to_dataframe(id2word: Any) -> pd.DataFrame:
    """Return token id to word mapping `id2word` as a pandas DataFrane, with DFS added"""

    assert id2word is not None, 'id2word is empty'

    dfs = list(id2word.dfs.values()) or 0 if hasattr(id2word, 'dfs') else 0

    token_ids, tokens = list(zip(*id2word.items()))

    dictionary: pd.DataFrame = pd.DataFrame({'token_id': token_ids, 'token': tokens, 'dfs': dfs}).set_index('token_id')[
        ['token', 'dfs']
    ]

    return dictionary


def compile_inferred_topics_data(
    topic_model: Any,
    corpus: Sparse2Corpus | VectorizedCorpus,
    id2word: dict,
    document_index: DocumentIndex,
    n_tokens: int = 200,
) -> InferredTopicsData:

    if isinstance(corpus, VectorizedCorpus):
        corpus: Sparse2Corpus = Sparse2Corpus(corpus.data)

    engine: ITopicModelEngine = get_engine_by_model_type(topic_model)
    dictionary: pd.DataFrame = _id2word_to_dataframe(id2word)
    topic_token_weights: pd.DataFrame = engine.get_topic_token_weights(dictionary, n_tokens=n_tokens)
    topic_token_overview: pd.DataFrame = engine.get_topic_token_overview(topic_token_weights, n_tokens=n_tokens)

    document_index: pd.DataFrame = add_document_terms_count(document_index, corpus)

    document_topic_weights: pd.DataFrame = predict_document_topics(
        topic_model,
        corpus,
        document_index=document_index,
        minimum_probability=0.001,
    )

    inferred_topics_data: InferredTopicsData = InferredTopicsData(
        document_index,
        dictionary,
        topic_token_weights,
        topic_token_overview,
        document_topic_weights,
    )
    return inferred_topics_data
