from typing import List, Set

import pandas as pd
from penelope.corpus import DocumentIndex, Token2Id, VectorizedCorpus
from penelope.type_alias import FilenameTokensTuples
from penelope.utility import strip_extensions

from ..interface import ContextOpts, CoOccurrenceComputeResult, CoOccurrenceError
from ..windows_utility import tokens_to_windows
from .convert import term_term_matrix_to_co_occurrences
from .vectorize import WindowsCoOccurrenceVectorizer


def compute_corpus_co_occurrence(
    stream: FilenameTokensTuples,
    *,
    token2id: Token2Id,
    document_index: DocumentIndex,
    context_opts: ContextOpts,
    global_threshold_count: int,
    ingest_tokens: bool = True,
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

    ignore_ids: Set[int] = None if not context_opts.ignore_padding else {token2id.id2token[context_opts.pad]}

    for filename, tokens in stream:

        if ingest_tokens:
            token2id.ingest(tokens)

        document_co_occurrences: pd.DataFrame = compute_document_co_occurrence(
            vectorizer=vectorizer,
            tokens=tokens,
            context_opts=context_opts,
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

    # FIXME value_n_t moved out of co_occurrences computation to application layer

    return CoOccurrenceComputeResult(
        co_occurrences=co_occurrences,
        token2id=token2id,
        document_index=document_index,
        token_window_counts=vectorizer.token_windows_counts,
    )


def compute_document_co_occurrence(
    vectorizer: WindowsCoOccurrenceVectorizer,
    tokens: List[str],
    context_opts: ContextOpts,
    ignore_ids: set,
) -> pd.DataFrame:

    windows = tokens_to_windows(tokens=tokens, context_opts=context_opts)
    windows_ttm_matrix: VectorizedCorpus = vectorizer.fit_transform(windows)
    co_occurrences: pd.DataFrame = term_term_matrix_to_co_occurrences(
        windows_ttm_matrix,
        threshold_count=1,
        ignore_ids=ignore_ids,
    )

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
