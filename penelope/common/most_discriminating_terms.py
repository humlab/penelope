import logging

import pandas as pd
import penelope.vendor.textacy.mdw_modified as mdw

logger = logging.getLogger(__name__)


def compute_most_discriminating_terms(
    x_corpus, x_documents: pd.DataFrame, top_n_terms: int = 25, max_n_terms: int = 1000, period1=None, period2=None
):

    group1_indices = x_documents[x_documents.year.between(*period1)].index
    group2_indices = x_documents[x_documents.year.between(*period2)].index

    if len(group1_indices) == 0 or len(group2_indices) == 0:
        return None

    indices = group1_indices.append(group2_indices)

    in_group1 = [True] * group1_indices.size + [False] * group2_indices.size

    dtm = x_corpus.data[indices, :]
    terms = mdw.most_discriminating_terms(
        dtm, x_corpus.id2token, in_group1, top_n_terms=top_n_terms, max_n_terms=max_n_terms
    )
    min_terms = min(len(terms[0]), len(terms[1]))
    df = pd.DataFrame({'Group 1': terms[0][:min_terms], 'Group 2': terms[1][:min_terms]})

    return df
