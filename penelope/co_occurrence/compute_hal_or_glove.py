import numpy as np
import pandas as pd
import penelope.utility as utility
import penelope.vendor.gensim as gensim_utility

from .vectorizer_glove import GloveVectorizer
from .vectorizer_hal import HyperspaceAnalogueToLanguageVectorizer

logger = utility.getLogger('corpus_text_analysis')


def compute(
    corpus,
    documents: pd.DataFrame,
    window_size: int,
    distance_metric: int,  # 0, 1, 2
    normalize: str = 'size',
    method: str = 'HAL',
    zero_diagonal: bool = True,
    direction_sensitive: bool = False,
):

    doc_terms = [[t.lower().strip('_') for t in terms if len(t) > 2] for terms in corpus.get_texts()]

    common_token2id = gensim_utility.build_vocab(doc_terms)

    dfs = []
    min_year, max_year = documents.year.min(), documents.year.max()
    documents['sequence_id'] = range(0, len(documents))

    for year in range(min_year, max_year + 1):

        year_indexes = list(documents.loc[documents.year == year].documents)

        docs = [doc_terms[y] for y in year_indexes]

        logger.info('Year %s...', year)

        if method == "HAL":

            vectorizer = HyperspaceAnalogueToLanguageVectorizer(token2id=common_token2id).fit(
                docs, size=window_size, distance_metric=distance_metric
            )

            df = vectorizer.cooccurence(
                direction_sensitive=direction_sensitive, normalize=normalize, zero_diagonal=zero_diagonal
            )

        else:

            vectorizer = GloveVectorizer(token2id=common_token2id).fit(docs, size=window_size)

            df = vectorizer.cooccurence(normalize=normalize, zero_diagonal=zero_diagonal)

        df['year'] = year
        # df = df[df.cwr >= threshhold]

        dfs.append(df[['year', 'x_term', 'y_term', 'nw_xy', 'nw_x', 'nw_y', 'cwr']])

        # if i == 5: break

    df = pd.concat(dfs, ignore_index=True)

    df['cwr'] = df.cwr / np.max(df.cwr, axis=0)

    return df
