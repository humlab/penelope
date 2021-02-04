from typing import Any

import gensim
import numpy as np
import pandas as pd
import penelope.utility as utility
from gensim.matutils import Sparse2Corpus
from penelope.corpus import DocumentIndex
from penelope.topic_modelling.container import InferredTopicsData
from penelope.topic_modelling.extract import extract_topic_token_overview, extract_topic_token_weights

from .utility import add_document_terms_count, id2word_to_dataframe

logger = utility.get_logger('corpus_text_analysis')


def predict_document_topics(
    model: Any,
    corpus: Any,
    document_index: pd.DataFrame = None,
    minimum_probability: float = 0.001,
) -> pd.DataFrame:
    """Applies a the topic model on `corpus` and returns a document-topic dataframe

    Parameters
    ----------
    model : ModelData
        The topic model
    corpus : Any
        The corpus
    document_index : pd.DataFrame, optional
        The document index, by default None
    minimum_probability : float, optional
        Threshold, by default 0.001

    Returns
    -------
    pd.DataFrame
        Document topics
    """
    try:

        def document_topics_iter(model, corpus, minimum_probability=0.0):

            if isinstance(model, gensim.models.LsiModel):
                # Gensim LSI Model
                data_iter = enumerate(model[corpus])
            elif hasattr(model, 'get_document_topics'):
                # Gensim LDA Model
                data_iter = enumerate(model.get_document_topics(corpus, minimum_probability=minimum_probability))
            elif hasattr(model, 'load_document_topics'):
                # Gensim MALLET wrapper
                # FIXME: Must do topic inference on corpus!
                data_iter = enumerate(model.load_document_topics())
            elif hasattr(model, 'top_doc_topics'):
                # scikit-learn, not that the corpus DTM is tored as a Gensim sparse corpus
                assert isinstance(corpus, Sparse2Corpus), "Only Sparse2Corpus valid for inference!"
                data_iter = model.top_doc_topics(corpus.sparse, docs=-1, top_n=1000, weights=True)
            else:
                data_iter = ((document_id, model[corpus[document_id]]) for document_id in range(0, len(corpus)))

                # assert False, 'compile_document_topics: Unknown topic model'

            for document_id, topic_weights in data_iter:
                for (topic_id, weight) in (
                    (topic_id, weight) for (topic_id, weight) in topic_weights if weight >= minimum_probability
                ):
                    yield (document_id, topic_id, weight)

        '''
        Get document topic weights for all documents in corpus
        Note!  minimum_probability=None filters less probable topics, set to 0 to retrieve all topcs

        If gensim model then use 'get_document_topics', else 'load_document_topics' for mallet model
        '''
        logger.info('Compiling document topics...')
        logger.info('  Creating data iterator...')
        data = document_topics_iter(model, corpus, minimum_probability)

        logger.info('  Creating frame from iterator...')
        df_doc_topics = pd.DataFrame(data, columns=['document_id', 'topic_id', 'weight'])

        df_doc_topics['document_id'] = df_doc_topics.document_id.astype(np.uint32)
        df_doc_topics['topic_id'] = df_doc_topics.topic_id.astype(np.uint16)

        df_doc_topics = DocumentIndex(document_index).overload(df_doc_topics, 'year')

        logger.info('  DONE!')

        return df_doc_topics

    except Exception as ex:  # pylint: disable=broad-except
        logger.error(ex)
        return None


def compile_inferred_topics_data(
    topic_model: Any, corpus: Any, id2word: Any, document_index: pd.DataFrame, n_tokens: int = 200
) -> InferredTopicsData:

    dictionary = id2word_to_dataframe(id2word)
    topic_token_weights = extract_topic_token_weights(topic_model, dictionary, n_tokens=n_tokens)
    topic_token_overview = extract_topic_token_overview(topic_model, topic_token_weights, n_tokens=n_tokens)

    document_index = add_document_terms_count(document_index, corpus)

    document_topic_weights = predict_document_topics(
        topic_model,
        corpus,
        document_index=document_index,
        minimum_probability=0.001,
    )

    inferred_topics_data = InferredTopicsData(
        document_index,
        dictionary,
        topic_token_weights,
        topic_token_overview,
        document_topic_weights,
    )
    return inferred_topics_data
