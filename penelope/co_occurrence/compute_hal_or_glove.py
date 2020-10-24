import numpy as np
import pandas as pd

import penelope.utility as utility
import penelope.vendor.gensim as gensim_utility

from .vectorizer_glove import GloveVectorizer
from .vectorizer_hal import HyperspaceAnalogueToLanguageVectorizer

logger = utility.getLogger('corpus_text_analysis')


def compute(
    corpus,
    document_index,
    window_size,
    distance_metric,
    normalize='size',
    method='HAL',
    zero_diagonal=True,
    direction_sensitive=False,
):

    doc_terms = [[t.lower().strip('_') for t in terms if len(t) > 2] for terms in corpus.get_texts()]

    common_token2id = gensim_utility.build_vocab(doc_terms)

    dfs = []
    min_year, max_year = document_index.year.min(), document_index.year.max()
    document_index['sequence_id'] = range(0, len(document_index))

    for year in range(min_year, max_year + 1):

        year_indexes = list(document_index.loc[document_index.year == year].sequence_id)

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
