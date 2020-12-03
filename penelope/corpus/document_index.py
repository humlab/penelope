import logging
from io import StringIO
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from penelope.utility import strip_path_and_extension


def assert_is_monotonic_increasing_integer_series(series: pd.Series):
    if not is_monotonic_increasing_integer_series(series):
        raise ValueError(f"series: {series.name} must be an integer typed, monotonic increasing series starting from 0")


def is_monotonic_increasing_integer_series(series: pd.Series):
    if not np.issubdtype(series.dtype, np.integer):
        return False
    if not series.is_monotonic_increasing:
        return False
    if len(series) > 0 and series.min() != 0:
        return False
    return True


def metadata_to_document_index(metadata: List[Dict], *, document_id_field: str = None) -> pd.DataFrame:
    """Creates a document index with all collected metadata key as columns.
    The data frame's index is set to `filename` by (default) or `index_field` if it is supplied.
    The index is also added as a `document_id` column."""

    if metadata is None or len(metadata) == 0:
        metadata = {'filename': [], 'document_id': []}

    catalogue: pd.DataFrame = pd.DataFrame(metadata)

    if 'filename' not in catalogue.columns:
        raise ValueError("metadata is missing mandatory field `filename`")

    if 'document_id' in catalogue.columns:
        logging.warning("filename metadata already has a column named `document_id` (will be overwritten)")

    if 'document_name' not in catalogue.columns:
        catalogue['document_name'] = catalogue.filename.apply(strip_path_and_extension)

    catalogue['document_id'] = catalogue[document_id_field] if document_id_field is not None else catalogue.index

    assert_is_monotonic_increasing_integer_series(catalogue['document_id'])

    catalogue = catalogue.set_index('filename', drop=False)

    return catalogue


def store_document_index(document_index: pd.DataFrame, filename: str):
    document_index.to_csv(filename, sep='\t', header=True)


def load_document_index(filename: Union[str, StringIO], *, key_column: str, sep: str) -> pd.DataFrame:
    """Loads a document index and sets `key_column` as index column. Also adds `document_id`"""

    attrs = dict(sep=sep)

    catalogue = pd.read_csv(filename, **attrs)

    if key_column is not None:
        if key_column not in catalogue.columns:
            raise ValueError(f"specified key column {key_column} not found in columns")

    if 'document_id' not in catalogue.columns and key_column is not None:
        catalogue['document_id'] = catalogue[key_column]
    else:
        catalogue['document_id'] = catalogue.index

    if 'document_name' not in catalogue.columns:
        catalogue['document_name'] = catalogue.filename.apply(strip_path_and_extension)

    assert_is_monotonic_increasing_integer_series(catalogue.document_id)

    catalogue = catalogue.set_index('filename', drop=False)

    return catalogue


def load_document_index_from_str(data_str: str, key_column: str, sep: str) -> pd.DataFrame:
    df = load_document_index(StringIO(data_str), key_column=key_column, sep=sep)
    return df


def consolidate_document_index(index: pd.DataFrame, reader_index: pd.DataFrame):
    """Returns a consolidated document index from an existing index, if exists,
    and the reader index."""

    if index is not None:
        columns = [x for x in reader_index.columns if x not in index.columns]
        index = index.merge(reader_index[columns], left_index=True, right_index=True, how='left')
        return index

    return reader_index


def document_index_upgrade(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Fixes older versions of document indexes"""

    if catalogue.index.dtype == np.dtype('int64'):

        if 'document_id' not in catalogue.columns:
            catalogue['document_id'] = catalogue.index

        catalogue = catalogue.set_index('filename', drop=False).rename_axis('')

    if 'document_name' not in catalogue.columns:
        catalogue['document_name'] = catalogue.filename.apply(strip_path_and_extension)

    return catalogue


def add_document_index_attributes(*, catalogue: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """ Adds document meta data to given data frame (must have a document_id) """
    df = target.merge(catalogue, how='inner', left_on='document_id', right_on='document_id')
    return df
