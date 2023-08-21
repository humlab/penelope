from dataclasses import dataclass

import pandas as pd

from penelope.utility import pandas_utils as pu

TEMPORAL_GROUP_BY = ['decade', 'lustrum', 'year']


@dataclass
class ComputeOpts:
    source_folder: str
    document_index: pd.DataFrame
    normalize: bool
    smooth: bool
    pos_groups: list[str]
    temporal_key: str
    pivot_keys_id_names: list[str] = None
    filter_opts: pu.PropertyValueMaskingOpts = None
    unstack_tabular: bool = None


def prepare_document_index(document_index: str, keep_columns: list[str]) -> pd.DataFrame:
    """Prepares document index by adding/renaming columns

    Args:
        source (str): document index source
        columns (list[str]): PoS-groups column names
    """

    if 'n_raw_tokens' not in document_index.columns:
        raise ValueError("expected required column `n_raw_tokens` not found")

    document_index['lustrum'] = document_index.year - document_index.year % 5
    document_index['decade'] = document_index.year - document_index.year % 10
    document_index = document_index.rename(columns={"n_raw_tokens": "Total"}).fillna(0)

    """strip away irrelevant columns"""
    groups = TEMPORAL_GROUP_BY + ['Total'] + sorted(keep_columns)
    keep_columns = [x for x in groups if x in document_index.columns]
    document_index = document_index[keep_columns]

    return document_index


def compute_statistics(document_index: pd.DataFrame, opts: ComputeOpts) -> pd.DataFrame:
    di: pd.DataFrame = document_index

    if opts.filter_opts is not None:
        di = opts.filter_opts.apply(di)

    pivot_keys: list[str] = [opts.temporal_key] + list(opts.pivot_keys_id_names)

    count_columns: list[str] = (
        list(opts.pos_groups)
        if len(opts.pos_groups or []) > 0
        else [x for x in di.columns if x not in TEMPORAL_GROUP_BY + ['Total'] + pivot_keys]
    )
    data: pd.DataFrame = di.groupby(pivot_keys).sum()[count_columns]

    if opts.normalize:
        total: pd.Series = di.groupby(pivot_keys)['Total'].sum()
        data = data.div(total, axis=0)

    # if opts.smooth:
    #     method: str = 'linear' if isinstance(data, pd.MultiIndex) else 'index'
    #     data = data.interpolate(method=method)

    data = data.reset_index()[pivot_keys + count_columns]
    return data
