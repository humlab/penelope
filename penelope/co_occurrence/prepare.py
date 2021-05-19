from typing import List

import pandas as pd
from pandas.api.types import is_numeric_dtype
from penelope.corpus import Id2Token


def prepare_tabular_data(
    co_occurrences: pd.DataFrame,
    id2token: Id2Token,
    threshold: int = 25,
    match_tokens: List[str] = None,
    skip_tokens: List[str] = None,
    n_head: int = 100000,
) -> pd.DataFrame:

    if len(co_occurrences) > n_head:
        print(f"warning: only {n_head} records out of {len(co_occurrences)} records are displayed.")

    for token in match_tokens or []:
        co_occurrences = co_occurrences[
            (co_occurrences.w1 == token)
            | (co_occurrences.w2 == token)
            | co_occurrences.w1.str.startswith(f"{token}@")
            | co_occurrences.w2.str.startswith(f"{token}@")
        ]

    for token in skip_tokens or []:
        co_occurrences = co_occurrences[
            (co_occurrences.w1 != token)
            & (co_occurrences.w2 != token)
            & ~co_occurrences.w1.str.startswith(f"{token}@")
            & ~co_occurrences.w2.str.startswith(f"{token}@")
        ]

    co_occurrences = co_occurrences.copy()

    co_occurrences["tokens"] = co_occurrences.w1 + "/" + co_occurrences.w2

    global_tokens_counts: pd.Series = co_occurrences.groupby(['tokens'])['value'].sum()
    threshold_tokens: pd.Index = global_tokens_counts[global_tokens_counts >= threshold].index

    co_occurrences = co_occurrences.set_index('tokens').loc[threshold_tokens]  # [['year', 'value', 'value_n_t']]

    if is_numeric_dtype(co_occurrences['value_n_t'].dtype):
        co_occurrences['value_n_t'] = co_occurrences.value_n_t.apply(lambda x: f'{x:.8f}')

    return co_occurrences.head(n_head)

