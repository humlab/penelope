import os
import pickle
import sys
import types
from typing import Any, Dict, List, Tuple

import gensim
import numpy as np
import pandas as pd

import penelope.utility as utility

logger = utility.getLogger('corpus_text_analysis')


class CompiledData():
    """Aggregate container for a topic model applied to a corpus and stored as pandas DataFrames.
    """

    def __init__(
        self, documents: pd.DataFrame, dictionary: Any, topic_token_weights: pd.DataFrame,
        topic_token_overview: pd.DataFrame, document_topic_weights: pd.DataFrame
    ):
        """
        Parameters
        ----------
        documents : pd.DataFrame
            Corpus document index
        dictionary : Any
            Corpus dictionary
        topic_token_weights : pd.DataFrame
            Topic token weights
        topic_token_overview : pd.DataFrame
            Topic overview
        document_topic_weights : pd.DataFrame
            Document topic weights
        """
        self.dictionary = dictionary
        self.documents = documents
        self.topic_token_weights = topic_token_weights
        self.topic_token_overview = topic_token_overview
        self.document_topic_weights = document_topic_weights

        # Ensure that `year` column exists

        self.document_topic_weights = extend_with_document_column(self.document_topic_weights, 'year', documents)

    @property
    def year_period(self) -> Tuple[int, int]:
        """Returns documents `year` interval (if exists)"""
        if 'year' not in self.document_topic_weights.columns:
            return (None, None)
        return (self.document_topic_weights.year.min(), self.document_topic_weights.year.max())

    @property
    def topic_ids(self) -> List[int]:
        """Returns unique topic ids """
        return list(self.document_topic_weights.topic_id.unique())

    def store(self, data_folder, model_name, pickled=False):
        """Stores aggregate in `data_folder` with filenames prefixed by `model_name`

        Parameters
        ----------
        data_folder : str
            target folder
        model_name : str
            Model name
        pickled : bool, optional
            if True then pickled (binary) format else  CSV, by default False
        """
        target_folder = os.path.join(data_folder, model_name)

        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)

        if pickled:

            filename = os.path.join(target_folder, "compiled_data.pickle")

            c_data = types.SimpleNamespace(
                documents=self.documents,
                dictionary=self.dictionary,
                topic_token_weights=self.topic_token_weights,
                topic_token_overview=self.topic_token_overview,
                document_topic_weights=self.document_topic_weights
            )
            with open(filename, 'wb') as f:
                pickle.dump(c_data, f, pickle.HIGHEST_PROTOCOL)

        else:

            self.documents.to_csv(os.path.join(target_folder, 'documents.zip'), '\t')
            self.dictionary.to_csv(os.path.join(target_folder, 'dictionary.zip'), '\t')
            self.topic_token_weights.to_csv(os.path.join(target_folder, 'topic_token_weights.zip'), '\t')
            self.topic_token_overview.to_csv(os.path.join(target_folder, 'topic_token_overview.zip'), '\t')
            self.document_topic_weights.to_csv(os.path.join(target_folder, 'document_topic_weights.zip'), '\t')

    @staticmethod
    def load(folder, pickled: bool = False):
        """Loads previously stored aggregate"""
        data = None

        if pickled:

            filename = os.path.join(folder, "compiled_data.pickle")

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            data = CompiledData(
                data.documents, data.dictionary, data.topic_token_weights, data.topic_token_overview,
                data.document_topic_weights
            )

        else:
            data = CompiledData(
                pd.read_csv(os.path.join(folder, 'documents.zip'), '\t', header=0, index_col=0, na_filter=False),
                pd.read_csv(os.path.join(folder, 'dictionary.zip'), '\t', header=0, index_col=0, na_filter=False),
                pd.read_csv(
                    os.path.join(folder, 'topic_token_weights.zip'), '\t', header=0, index_col=0, na_filter=False
                ),
                pd.read_csv(
                    os.path.join(folder, 'topic_token_overview.zip'), '\t', header=0, index_col=0, na_filter=False
                ),
                pd.read_csv(
                    os.path.join(folder, 'document_topic_weights.zip'), '\t', header=0, index_col=0, na_filter=False
                )
            )

        return data

    def info(self):
        for o_name in [k for k in self.__dict__ if not k.startswith("__")]:
            o_data = getattr(self, o_name)
            o_size = sys.getsizeof(o_data)
            print('{:>20s}: {:.4f} Mb {}'.format(o_name, o_size / (1024 * 1024), type(o_data)))

    @property
    def id2term(self):
        return self.dictionary.token.to_dict()

    @property
    def term2id(self):
        return {v: k for k, v in self.id2term.items()}


def id2word2dataframe(id2word: Dict) -> pd.DataFrame:
    """Returns token id to word mapping `id2word` as a pandas DataFrane, with DFS added

    Parameters
    ----------
    id2word : Dict
        Token ID to word mapping

    Returns
    -------
    pd.DataFrame
        dictionary as dataframe
    """
    logger.info('Compiling dictionary...')

    assert id2word is not None, 'id2word is empty'

    dfs = list(id2word.dfs.values()) or 0 if hasattr(id2word, 'dfs') else 0

    token_ids, tokens = list(zip(*id2word.items()))

    dictionary = pd.DataFrame({
        'token_id': token_ids,
        'token': tokens,
        'dfs': dfs
    }).set_index('token_id')[['token', 'dfs']]

    return dictionary


def _compile_topic_token_weights(
    model, dictionary: pd.DataFrame, n_tokens: int = 200, minimum_probability: float = 0.000001
) -> pd.DataFrame:
    """Creates a DataFrame containing document topic weights

    Parameters
    ----------
    model : [type]
        The topic model
    dictionary : pd.DataFrame
        The ID to word mapping
    n_tokens : int, optional
        Number of tokens to include per topic, by default 200
    minimum_probability : float, optional
        Minimum probability consider, by default 0.000001

    Returns
    -------
    [type]
        [description]
    """
    logger.info('Compiling topic-tokens weights...')

    id2term = dictionary.token.to_dict()
    term2id = {v: k for k, v in id2term.items()}

    if hasattr(model, 'show_topics'):
        # Gensim LDA model
        topic_data = model.show_topics(num_topics=-1, num_words=n_tokens, formatted=False)
    elif hasattr(model, 'top_topic_terms'):
        # Textacy/scikit-learn model
        topic_data = model.top_topic_terms(id2term, topics=-1, top_n=n_tokens, weights=True)
    else:
        assert False, "Unknown model type"

    df_topic_weights = pd.DataFrame([(topic_id, token, weight) for topic_id, tokens in topic_data
                                     for token, weight in tokens if weight > minimum_probability],
                                    columns=['topic_id', 'token', 'weight'])

    df_topic_weights['topic_id'] = df_topic_weights.topic_id.astype(np.uint16)

    term2id = {v: k for k, v in id2term.items()}
    df_topic_weights['token_id'] = df_topic_weights.token.apply(lambda x: term2id[x])

    return df_topic_weights[['topic_id', 'token_id', 'token', 'weight']]


def _compile_topic_token_overview(
    topic_token_weights: pd.DataFrame, alpha: List[float] = None, n_tokens: int = 200
) -> pd.DataFrame:
    """
    Group by topic_id and concatenate n_tokens words within group sorted by weight descending.
    There must be a better way of doing this...
    """
    logger.info('Compiling topic-tokens overview...')

    df = topic_token_weights.groupby('topic_id')\
        .apply(lambda x: sorted(list(zip(x["token"], x["weight"])), key=lambda z: z[1], reverse=True))\
        .apply(lambda x: ' '.join([z[0] for z in x][:n_tokens])).reset_index()
    df.columns = ['topic_id', 'tokens']
    df['alpha'] = df.topic_id.apply(lambda topic_id: alpha[topic_id]) if alpha is not None else 0.0

    return df.set_index('topic_id')


def _compile_document_topics(
    model,
    corpus: Any,
    documents: pd.DataFrame = None,
    doc_topic_matrix: Any = None,
    minimum_probability: float = 0.001
) -> pd.DataFrame:
    """Compiles a document topic dataframe for `corpus`

    Parameters
    ----------
    model : ModelData
        The topic model
    corpus : Any
        The corpus
    documents : pd.DataFrame, optional
        The document index, by default None
    doc_topic_matrix : Any, optional
        The document-topic sparse matrix, by default None
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
                data_iter = enumerate(model.load_document_topics())
            elif hasattr(model, 'top_doc_topics'):
                # scikit-learn
                assert doc_topic_matrix is not None, "doc_topic_matrix not supplied"
                data_iter = model.top_doc_topics(doc_topic_matrix, docs=-1, top_n=1000, weights=True)
            else:
                data_iter = ((document_id, model[corpus[document_id]]) for document_id in range(0, len(corpus)))

                # assert False, 'compile_document_topics: Unknown topic model'

            for document_id, topic_weights in data_iter:
                for (topic_id, weight) in ((topic_id, weight) for (topic_id, weight) in topic_weights
                                           if weight >= minimum_probability):
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

        df_doc_topics = extend_with_document_column(df_doc_topics, 'year', documents)

        logger.info('  DONE!')

        return df_doc_topics

    except Exception as ex:  # pylint: disable=broad-except
        logger.error(ex)
        return None


def compile_data(model: Any, corpus: Any, id2term, documents, doc_topic_matrix=None, n_tokens=200):
    """Main function. Returns compiled aggregated topic modelling data when model is applied to corpus

    Parameters
    ----------
    model : ModelData
        The topic model
    corpus : Any
        The corpus
    id2term : Any
        The corpus
    documents : pd.DataFrame, optional
        The document index, by default None
    doc_topic_matrix : Any, optional
        The document-topic sparse matrix, by default None
    n_tokens : float, optional
        Number of tokens, by default 0.001

    Returns
    -------
    CompiledData
        Compiled aggregated topic modelling data
    """

    try:
        """ Fix missing n_terms (only vectorized corps"""
        if 'n_terms' not in documents.columns:

            if hasattr(corpus, 'sparse'):
                documents['n_terms'] = corpus.sparse.sum(axis=0).A1

            if isinstance(corpus, list):
                documents['n_terms'] = [sum((w[1] for w in d)) for d in corpus]

        dictionary = id2word2dataframe(id2term)
        topic_token_weights = _compile_topic_token_weights(model, dictionary, n_tokens=n_tokens)
        alpha = model.alpha if 'alpha' in model.__dict__ else None
        topic_token_overview = _compile_topic_token_overview(topic_token_weights, alpha)

        document_topic_weights = _compile_document_topics(
            model, corpus, documents=documents, doc_topic_matrix=doc_topic_matrix, minimum_probability=0.001
        )

        data = CompiledData(documents, dictionary, topic_token_weights, topic_token_overview, document_topic_weights)

        return data

    except Exception as ex:  # pylint: disable=broad-except
        logger.exception(ex)
        return None


def extend_with_document_column(df, column, documents):
    """Add document `column` to `df` if not already exists (and year exists in documents).

    Parameters
    ----------
    df : DataFrame
        The dataframe to extend
    documents : DataFrame, optional
        Corpus document index, by default None

    Returns
    -------
    DataFrame
        `df` extended with `column`
    """

    if column in df.columns:
        return df

    if documents is None:
        return df

    if column not in documents.columns:
        return df

    if 'document_id' not in df.columns:
        return df

    df = df.merge(documents[column], how='inner', left_on='document_id', right_index=True)

    return df


def get_topic_titles(topic_token_weights: pd.DataFrame, topic_id: int = None, n_tokens: int = 100) -> pd.DataFrame:
    """Returns a DataFrame containing a string of `n_tokens` most probable words per topic"""

    df_temp = topic_token_weights if topic_id is None else topic_token_weights[
        (topic_token_weights.topic_id == topic_id)]

    df = df_temp\
            .sort_values('weight', ascending=False)\
            .groupby('topic_id')\
            .apply(lambda x: ' '.join(x.token[:n_tokens].str.title()))

    return df


def get_topic_title(topic_token_weights: pd.DataFrame, topic_id: int, n_tokens: int = 100) -> str:
    """Returns a string of `n_tokens` most probable words per topic"""
    return get_topic_titles(topic_token_weights, topic_id, n_tokens=n_tokens).iloc[0]


def get_topic_tokens(topic_token_weights: pd.DataFrame, topic_id: int = None, n_tokens: int = 100) -> pd.DataFrame:
    """Returns most probable tokens for given topic sorted by probability descending"""
    df_temp = topic_token_weights if topic_id is None else topic_token_weights[
        (topic_token_weights.topic_id == topic_id)]
    df = df_temp.sort_values('weight', ascending=False)[:n_tokens]
    return df


def get_topics_unstacked(
    model, n_tokens: int = 20, id2term: Dict[int, str] = None, topic_ids: List[int] = None
) -> pd.DataFrame:
    """Returns the top `n_tokens` tokens for each topic. The token's column index is in ascending probability"""

    if hasattr(model, 'num_topics'):
        # Gensim LDA model
        show_topic = lambda topic_id: model.show_topic(topic_id, topn=n_tokens)
        n_topics = model.num_topics
    elif hasattr(model, 'm_T'):
        # Gensim HDP model
        show_topic = lambda topic_id: model.show_topic(topic_id, topn=n_tokens)
        n_topics = model.m_T
    else:
        # Textacy/scikit-learn model
        def scikit_learn_show_topic(topic_id):
            topic_words = list(model.top_topic_terms(id2term, topics=(topic_id, ), top_n=n_tokens, weights=True))
            if len(topic_words) == 0:
                return []
            return topic_words[0][1]

        show_topic = scikit_learn_show_topic
        n_topics = model.n_topics

    topic_ids = topic_ids or range(n_topics)

    return pd.DataFrame({
        'Topic#{:02d}'.format(topic_id + 1): [word[0] for word in show_topic(topic_id)]
        for topic_id in topic_ids
    })
