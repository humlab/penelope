import logging
from io import StringIO
from typing import Dict, List, Union

import numpy as np
import pandas as pd


# FIXME: Move closer to to load_documents since matching logic
def metadata_to_document_index(metadata: List[Dict], *, document_id_field: str = None) -> pd.DataFrame:
    """Creates a document index with all collected metadata key as columns.
    The data frame's index is set to `filename` by (default) or `index_field` if it is supplied.
    The index is also added as a `document_id` column."""

    if metadata is None or len(metadata) == 0:
        metadata = {'filename': [], 'document_id': []}

    document_index: pd.DataFrame = pd.DataFrame(metadata)

    if 'filename' not in document_index.columns:
        raise ValueError("metadata is missing mandatory field `filename`")

    if 'document_id' in document_index.columns:
        logging.warning("filename metadata already has a column named `document_id` (will be overwritten)")

    if 'document_name' not in document_index.columns:
        document_index['document_name'] = document_index['filename']

    document_index['document_id'] = (
        document_index[document_id_field] if not document_id_field is None else document_index.index
    )
    document_index = document_index.set_index('filename', drop=False)

    return document_index


def store_document_index(document_index: pd.DataFrame, filename: str):
    document_index.to_csv(filename, sep='\t', header=True)


def load_document_index(filename: Union[str, StringIO], *, key_column: str, sep: str) -> pd.DataFrame:
    """Loads a document index and sets `key_column` as index column. Also adds `document_id`"""

    attrs = dict(sep=sep)
    # if key_column is None:
    #     attrs['index_col'] = 0

    df = pd.read_csv(filename, **attrs)

    if key_column is not None:
        if key_column not in df.columns:
            raise ValueError(f"specified key column {key_column} not found in columns")

    if 'document_id' not in df.columns:
        df['document_id'] = df[key_column]
    else:
        df['document_id'] = df.index

    if 'document_name' not in df.columns:
        df['document_name'] = df['filename']

    df = df.set_index('filename', drop=False)

    return df


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


def document_index_upgrade(documents: pd.DataFrame) -> pd.DataFrame:
    """Fixes older versions of document indexes"""

    if documents.index.dtype == np.dtype('int64'):

        if 'document_id' not in documents.columns:
            documents['document_id'] = documents.index

        documents = documents.set_index('filename', drop=False).rename_axis('')

    if 'document_name' not in documents.columns:
        documents['document_name'] = documents.filename

    return documents
