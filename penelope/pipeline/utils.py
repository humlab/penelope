from io import StringIO
from typing import Iterable, Union

import pandas as pd


def to_text(data: Union[str, Iterable[str]]):
    return data if isinstance(data, str) else ' '.join(data)


# def read_data_frame_from_zip(zf, filename):
#     data_str = zf.read(filename).decode('utf-8')
#     data_source = StringIO(data_str)
#     df = pd.read_csv(data_source, sep='\t', index_col=0)
#     return df


# def write_data_frame_to_zip(df: pd.DataFrame, filename: str, zf: zipfile.ZipFile):
#     assert isinstance(df, (pd.DataFrame,))
#     data_str: str = df.to_csv(sep='\t', header=True)
#     zf.writestr(filename, data=data_str)


def store_document_index(document_index: pd.DataFrame, filename: str):
    document_index.to_csv(filename, sep='\t', header=True)


def load_document_index(filename: Union[str, StringIO], *, key_column: str, sep: str) -> pd.DataFrame:
    """Loads a document index and sets `key_column` as index column. Also adds `document_id`"""

    attrs = dict(sep=sep)
    if key_column is None:
        attrs['index_col'] = 0

    df = pd.read_csv(filename, **attrs)

    if key_column is not None:
        if key_column not in df.columns:
            raise ValueError(f"specified key column {key_column} not found in columns")
        df = df.set_index(key_column, drop=False)

    if 'document_id' not in df.columns:
        df['document_id'] = df.index

    return df


def load_document_index_from_str(data_str: str, key_column: str, sep: str) -> pd.DataFrame:
    df = load_document_index(StringIO(data_str), key_column=key_column, sep=sep)
    return df


def consolidate_document_index(index: pd.DataFrame, reader_index: pd.DataFrame):
    """Returns a consolidated document index from ax existing index, if exists,
    and index created by reader."""

    if index is not None:
        columns = [x for x in reader_index.columns if x not in index.columns]
        index = index.merge(reader_index[columns], left_index=True, right_index=True, how='left')
        return index

    return reader_index
