from typing import List

import numpy as np
import pandas as pd
import scipy
from penelope.corpus import DocumentIndex, Token2Id, VectorizedCorpus
from penelope.type_alias import FilenameTokensTuples
from penelope.utility import strip_extensions

from ..interface import ContextOpts, CoOccurrenceComputeResult, CoOccurrenceError
from ..windows_utility import tokens_to_windows
from .vectorize import WindowsCoOccurrenceVectorizer


def compute_corpus_co_occurrence(
    stream: FilenameTokensTuples,
    *,
    token2id: Token2Id,
    document_index: DocumentIndex,
    context_opts: ContextOpts,
    global_threshold_count: int,
    ingest_tokens: bool = True,
    ignore_pad: bool = None,
) -> CoOccurrenceComputeResult:

    if token2id is None:
        raise CoOccurrenceError("expected `token2id` found None")

    if document_index is None:
        raise CoOccurrenceError("expected document index found None")

    if 'n_tokens' not in document_index.columns:
        raise CoOccurrenceError("expected `document_index.n_tokens`, but found no column")

    if 'n_raw_tokens' not in document_index.columns:
        raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no column")

    if not isinstance(global_threshold_count, int) or global_threshold_count < 1:
        global_threshold_count = 1

    if not isinstance(token2id, Token2Id):
        token2id = Token2Id(data=token2id)

    total_results = []

    vectorizer: WindowsCoOccurrenceVectorizer = WindowsCoOccurrenceVectorizer(token2id)

    for filename, tokens in stream:

        if ingest_tokens:
            token2id.ingest(tokens)

        document_co_occurrences: pd.DataFrame = compute_document_co_occurrence(
            vectorizer=vectorizer,
            tokens=tokens,
            context_opts=context_opts,
        )

        document_co_occurrences['document_id'] = document_index.loc[strip_extensions(filename)]['document_id']

        total_results.append(document_co_occurrences)

    co_occurrences: pd.DataFrame = pd.concat(total_results, ignore_index=True)[
        ['document_id', 'w1_id', 'w2_id', 'value']
    ]

    if ignore_pad:
        pad_id: int = token2id.id2token[context_opts.pad]
        co_occurrences = co_occurrences[((co_occurrences.w1_id != pad_id) & (co_occurrences.w2_id != pad_id))]

    if len(co_occurrences) > 0 and global_threshold_count > 1:
        co_occurrences = co_occurrences[
            co_occurrences.groupby(["w1_id", "w2_id"])['value'].transform('sum') >= global_threshold_count
        ]

    # FIXME: Don't add tokens - postpone to application layer
    # FIXME value_n_t moved out of co_occurrences computation to application layer

    return CoOccurrenceComputeResult(
        co_occurrences=co_occurrences,
        token2id=token2id,
        document_index=document_index,
        token_window_counts=vectorizer.token_windows_counts,
    )


def compute_document_co_occurrence(
    vectorizer: WindowsCoOccurrenceVectorizer, tokens: List[str], context_opts: ContextOpts
) -> pd.DataFrame:

    windows = tokens_to_windows(tokens=tokens, context_opts=context_opts)
    windows_ttm_matrix: VectorizedCorpus = vectorizer.fit_transform(windows)
    co_occurrences: pd.DataFrame = term_term_matrix_to_dataframe(windows_ttm_matrix, threshold_count=1)
    return co_occurrences


def term_term_matrix_to_dataframe(
    term_term_matrix: scipy.sparse.spmatrix,
    threshold_count: int = 1,
) -> pd.DataFrame:
    """Converts a TTM to a Pandas DataFrame

    Args:
        term_term_matrix (scipy.sparse.spmatrix): [description]
        threshold_count (int, optional): min threshold for global token count. Defaults to 1.

    Returns:
        pd.DataFrame: co-occurrence data frame
    """

    co_occurrences = (
        pd.DataFrame(
            {
                'w1_id': pd.Series(term_term_matrix.row, dtype=np.uint32),
                'w2_id': pd.Series(term_term_matrix.col, dtype=np.uint32),
                'value': term_term_matrix.data,
            },
        )
        .sort_values(['w1_id', 'w2_id'])
        .reset_index(drop=True)
    )

    if threshold_count > 1:
        co_occurrences = co_occurrences[co_occurrences.value >= threshold_count]

    return co_occurrences


# def compute_value_n_t(co_occurrences: pd.DataFrame, document_index: DocumentIndex):
#     if document_index is None:
#         return co_occurrences
#     for n_token_count, target_field in [('n_tokens', 'value_n_t'), ('n_raw_tokens', 'value_n_r_t')]:
#         if n_token_count in document_index.columns:
#             try:
#                 co_occurrences[target_field] = co_occurrences.value / float(sum(document_index[n_token_count].values))
#             except ZeroDivisionError:
#                 co_occurrences[target_field] = 0.0
#         else:
#             logger.warning(f"{target_field}: cannot compute since {n_token_count} not in corpus document catalogue")
