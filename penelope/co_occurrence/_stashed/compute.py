from typing import Set

import pandas as pd
import scipy
from penelope.corpus import DocumentIndex, Token2Id
from penelope.type_alias import FilenameTokensTuples
from penelope.utility import strip_extensions

from ..convert import term_term_matrix_to_co_occurrences
from ..interface import ContextOpts, CoOccurrenceComputeResult, CoOccurrenceError
from ..vectorize import WindowsCoOccurrenceVectorizer
from ..windows import tokens_to_windows


def compute_corpus_co_occurrence(
    stream: FilenameTokensTuples,
    *,
    token2id: Token2Id,
    document_index: DocumentIndex,
    context_opts: ContextOpts,
    global_threshold_count: int,
    ingest_tokens: bool = True,
) -> CoOccurrenceComputeResult:
    """Note: This function is currently ONLY used in test cases!"""
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

    ignore_ids: Set[int] = None if not context_opts.ignore_padding else {token2id.id2token[context_opts.pad]}

    for filename, tokens in stream:

        if ingest_tokens:
            token2id.ingest(tokens)

        windows = tokens_to_windows(tokens=tokens, context_opts=context_opts)
        windows_ttm_matrix: scipy.sparse.spmatrix = vectorizer.fit_transform(windows)
        document_co_occurrences = term_term_matrix_to_co_occurrences(
            windows_ttm_matrix,
            threshold_count=1,
            ignore_ids=ignore_ids,
        )

        document_co_occurrences['document_id'] = document_index.loc[strip_extensions(filename)]['document_id']

        total_results.append(document_co_occurrences)

    co_occurrences: pd.DataFrame = pd.concat(total_results, ignore_index=True)[
        ['document_id', 'w1_id', 'w2_id', 'value']
    ]

    if len(co_occurrences) > 0 and global_threshold_count > 1:
        co_occurrences = co_occurrences[
            co_occurrences.groupby(["w1_id", "w2_id"])['value'].transform('sum') >= global_threshold_count
        ]

    return CoOccurrenceComputeResult(
        co_occurrences=co_occurrences,
        token2id=token2id,
        document_index=document_index,
        token_window_counts=vectorizer.global_token_windows_counts,
    )
