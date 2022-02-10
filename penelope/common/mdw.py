import logging
from typing import Tuple

import pandas as pd

from penelope.corpus import dtm
from penelope.vendor import textacy_api

logger = logging.getLogger(__name__)


def compute_most_discriminating_terms(
    corpus: dtm.VectorizedCorpus,
    top_n_terms: int = 25,
    max_n_terms: int = 1000,
    period1: Tuple[int, int] = None,
    period2: Tuple[int, int] = None,
) -> pd.DataFrame:

    group1_indices = corpus.document_index[corpus.document_index.year.between(*period1)].index
    group2_indices = corpus.document_index[corpus.document_index.year.between(*period2)].index

    df_mdt = textacy_api.compute_most_discriminating_terms(
        corpus,
        group1_indices=group1_indices,
        group2_indices=group2_indices,
        top_n_terms=top_n_terms,
        max_n_terms=max_n_terms,
    )

    return df_mdt
