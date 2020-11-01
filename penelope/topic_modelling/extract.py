from typing import Any, List

import numpy as np
import pandas as pd
import penelope.utility as utility

logger = utility.getLogger('corpus_text_analysis')


def extract_topic_token_weights(
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

    df_topic_weights = pd.DataFrame(
        [
            (topic_id, token, weight)
            for topic_id, tokens in topic_data
            for token, weight in tokens
            if weight > minimum_probability
        ],
        columns=['topic_id', 'token', 'weight'],
    )

    df_topic_weights['topic_id'] = df_topic_weights.topic_id.astype(np.uint16)

    term2id = {v: k for k, v in id2term.items()}
    df_topic_weights['token_id'] = df_topic_weights.token.apply(lambda x: term2id[x])

    return df_topic_weights[['topic_id', 'token_id', 'token', 'weight']]


def extract_topic_token_overview(model: Any, topic_token_weights: pd.DataFrame, n_tokens: int = 200) -> pd.DataFrame:
    """
    Group by topic_id and concatenate n_tokens words within group sorted by weight descending.
    There must be a better way of doing this...
    """
    logger.info('Compiling topic-tokens overview...')

    alpha: List[float] = model.alpha if 'alpha' in model.__dict__ else None

    df = (
        topic_token_weights.groupby('topic_id')
        .apply(lambda x: sorted(list(zip(x["token"], x["weight"])), key=lambda z: z[1], reverse=True))
        .apply(lambda x: ' '.join([z[0] for z in x][:n_tokens]))
        .reset_index()
    )
    df.columns = ['topic_id', 'tokens']
    df['alpha'] = df.topic_id.apply(lambda topic_id: alpha[topic_id]) if alpha is not None else 0.0

    return df.set_index('topic_id')
