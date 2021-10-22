from typing import Any, Dict

import pandas as pd
import penelope.utility as utility

logger = utility.getLogger('corpus_text_analysis')

pd.options.mode.chained_assignment = None


def filter_by_key_value(df: pd.DataFrame, filters: Dict[str, Any] = None) -> pd.DataFrame:
    """Returns filtered dataframe based custom key/value equality filters`.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to be filtered
    filters : Dict[str, Any], optional
        [description], by default None

    Returns
    -------
    pd.DataFrame
        [description]
    """
    for k, v in (filters or {}).items():
        if k not in df.columns:
            logger.warning('Column %s does not exist in dataframe (filter_by_key_value)', k)
            continue
        df = df[df[k] == v]

    return df


def filter_document_topic_weights(
    document_topic_weights: pd.DataFrame, filters: Dict[str, Any] = None, threshold: float = 0.0
) -> pd.DataFrame:
    """Returns document's topic weights for given `year`, `topic_id`, custom `filters` and threshold.

    Parameters
    ----------
    document_topic_weights : pd.DataFrame
        Document topic weights
    filters : Dict[str, Any], optional
        [description], by default None
    threshold : float, optional
        [description], by default 0.0

    Returns
    -------
    pd.DataFrame
        [description]
    """
    df = document_topic_weights

    df = df[df.weight >= threshold]

    for k, v in (filters or {}).items():
        if k not in df.columns:
            logger.warning('Column %s does not exist in dataframe (_find_documents_for_topics)', k)
            continue
        df = df[df[k] == v]

    return df.copy()


def reduce_topic_tokens_overview(topics: pd.DataFrame, n_count: int, search: str) -> pd.DataFrame:

    reduced_topics = topics[topics.tokens.str.contains(search)] if search else topics

    tokens: pd.Series = reduced_topics.tokens.apply(lambda x: " ".join(x.split()[:n_count]))

    if search:
        tokens = reduced_topics.tokens.apply(
            lambda x: x.replace(search, f'<b style="color:green;font-size:14px">{search}</b>')
        )

    reduced_topics['tokens'] = tokens

    return reduced_topics
