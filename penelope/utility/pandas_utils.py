from __future__ import annotations

import fnmatch
import operator
import zipfile
from io import StringIO
from numbers import Number
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from .filename_utils import replace_extension
from .utils import now_timestamp

DataFrameFilenameTuple = tuple[pd.DataFrame, str]


def unstack_data(data: pd.DataFrame, pivot_keys: list[str]) -> pd.DataFrame:
    """Unstacks a dataframe that has been grouped by temporal_key and pivot_keys"""
    if len(pivot_keys) <= 1 or data is None:
        return data
    data: pd.DataFrame = data.set_index(pivot_keys)
    while isinstance(data.index, pd.MultiIndex):
        data = data.unstack(level=1, fill_value=0)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [' '.join(x) for x in data.columns]
    return data


def faster_to_dict_records(df: pd.DataFrame) -> list[dict]:
    data: list[Any] = df.values.tolist()
    columns: list[str] = df.columns.tolist()
    return [dict(zip(columns, datum)) for datum in data]


def set_default_options():
    pd.options.display.colheader_justify = 'left'
    pd.options.display.width = 1000
    pd.options.display.max_colwidth = 300
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None


def _create_mask(df: pd.DataFrame, name: str, value: Any, sign: bool = True) -> np.ndarray:
    if isinstance(
        value,
        (
            list,
            set,
        ),
    ):
        m = df[name].isin(value)
    elif isinstance(value, tuple):
        m = df[name].between(*value)
    elif isinstance(value, (bool, Number, str)):
        m = df[name] == value
    else:
        m = df[name] == value
    if not sign:
        m = ~m
    return m


def create_mask2(df: pd.DataFrame, masks: Sequence[dict]) -> np.ndarray:
    v = np.repeat(True, len(df.index))
    for m in masks:
        v &= _create_mask(df, **m)
    return v


class CreateMaskError(Exception):
    def __init__(self):
        super().__init__(
            """
        tuple length must be 2 or 3 and first element must be sign, second (optional) a binary op.
    """
        )


def size_of(df: pd.DataFrame, unit: Literal['bytes', 'kB', 'MB', 'GB'], total: bool = False) -> int | dict:
    d: dict = {x: 1024**i for i, x in enumerate(['bytes', 'kB', 'MB', 'GB'])}
    sizes: pd.Series = df.memory_usage(index=True, deep=True)
    return (
        f"{sizes.sum()/d[unit]:.1f} {unit}"
        if total
        else {k: f"{v/d[unit]:.1f} {unit}" for k, v in sizes.to_dict().items()}
    )


def create_mask(doc: pd.DataFrame, args: dict) -> np.ndarray:
    """Creates a mask based on key-values in `criterias`

    Each key-value in `criterias` specifies a filter that are combined using boolean `and`.

    Args:
        doc (pd.DataFrame): Data frame to mask
        criterias (dict): dict with masking criterias

        Filter applied on `df` for key-value (k, v):

            when value is (bool, fx, v)                 [bool] fx(df.k, attr_value)   fx, callable or string (i.e operator.fx)
                                (fx, v)                 fx(df.k, attr_value)
                                     v: list            df.k.isin(lst)
                                     v: set             df.k.isin(set)
                                     v                  df.k == v

    """
    mask = np.repeat(True, len(doc.index))

    if len(doc) == 0:
        return mask

    for attr_name, attr_value in args.items():
        attr_sign = True
        attr_operator: str | Callable = None

        if attr_value is None:
            continue

        if attr_name not in doc.columns:
            continue

        if isinstance(attr_value, tuple):
            if len(attr_value) not in (2, 3):
                raise CreateMaskError()

            if len(attr_value) == 3:
                attr_sign, attr_operator, attr_value = attr_value
            else:
                if isinstance(attr_value[0], bool):
                    attr_sign, attr_value = attr_value
                elif callable(attr_value[0]) or isinstance(attr_value[0], str):
                    attr_operator, attr_value = attr_value
                # else assume numric range (between)

            if isinstance(attr_operator, str):
                if not hasattr(operator, attr_operator):
                    raise ValueError(f"operator.{attr_operator} not found")

                attr_operator = getattr(operator, attr_operator)

        value_serie: pd.Series = doc[attr_name]

        attr_mask = (
            value_serie.between(*attr_value)
            if isinstance(attr_value, tuple)
            else attr_operator(value_serie, attr_value)
            if attr_operator is not None
            else value_serie.isin(attr_value)
            if isinstance(attr_value, (list, set))
            else value_serie == attr_value
        )

        if attr_sign:
            mask &= attr_mask
        else:
            mask &= ~attr_mask

    return mask


class PropertyValueMaskingOpts:
    """A simple key-value filter that returns a mask set to True for items that fulfills all criterias"""

    def __init__(self, **kwargs):
        super().__setattr__('data', kwargs or {})

    def __getitem__(self, key: int):
        return self.data[key]

    def __setitem__(self, k, v):
        self.data[k] = v

    def __len__(self):
        return len(self.data)

    def __setattr__(self, k, v):
        self.data[k] = v

    def __getattr__(self, k):
        try:
            return self.data[k]
        except KeyError:
            return None

    def __eq__(self, other: PropertyValueMaskingOpts) -> bool:
        if not isinstance(other, PropertyValueMaskingOpts):
            return False
        return self.data == other.props

    @property
    def props(self) -> dict:
        return self.data

    @property
    def opts(self) -> dict:
        return self.data

    def mask(self, doc: pd.DataFrame) -> np.ndarray:
        return create_mask(doc, self.data)

    def apply(self, doc: pd.DataFrame) -> pd.DataFrame:
        if len(self.hot_attributes(doc)) == 0:
            return doc
        return doc[self.mask(doc)]

    def hot_attributes(self, doc: pd.DataFrame) -> list[str]:
        """Returns attributes that __might__ filter tagged frame"""
        return [
            (attr_name, attr_value)
            for attr_name, attr_value in self.data.items()
            if attr_name in doc.columns and attr_value is not None
        ]

    @property
    def clone(self) -> PropertyValueMaskingOpts:
        return PropertyValueMaskingOpts(**self.props)

    def update(self, other: PropertyValueMaskingOpts | dict = None, **kwargs) -> PropertyValueMaskingOpts:
        if isinstance(other, dict):
            self.data.update(other)
        if kwargs:
            self.data.update(kwargs)
        if isinstance(other, PropertyValueMaskingOpts):
            self.data.update(other.data)
        return self


def try_split_column(
    df: pd.DataFrame,
    source_name: str,
    sep: str,
    target_names: list[str],
    drop_source: bool = True,
    probe_size: int = 10,
) -> pd.DataFrame:
    if df is None or len(df) == 0 or source_name not in df.columns:
        return df

    if probe_size > 0 and not df.head(probe_size)[source_name].str.match(rf".+{sep}\w+").all():
        return df

    df[target_names] = df[source_name].str.split(sep, n=1, expand=True)

    if source_name not in target_names and drop_source:
        df.drop(columns=source_name, inplace=True)

    return df


def pandas_to_csv_zip(
    zip_filename: str, dfs: DataFrameFilenameTuple | list[DataFrameFilenameTuple], extension='csv', **to_csv_opts
):
    if not isinstance(dfs, (list, tuple)):
        raise ValueError("expected tuple or list of tuples")

    if isinstance(dfs, (tuple,)):
        dfs = [dfs]

    with zipfile.ZipFile(zip_filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for df, filename in dfs:
            if not isinstance(df, pd.core.frame.DataFrame) or not isinstance(filename, str):
                raise ValueError(
                    f"Expected tuple[pd.DateFrame, filename: str], found tuple[{type(df)}, {type(filename)}]"
                )
            filename = replace_extension(filename=filename, extension=extension)
            data_str = df.to_csv(**to_csv_opts)
            zf.writestr(filename, data=data_str)


def pandas_read_csv_zip(zip_filename: str, pattern='*.csv', **read_csv_opts) -> dict:
    data = dict()
    with zipfile.ZipFile(zip_filename, mode='r') as zf:
        for filename in zf.namelist():
            if not fnmatch.fnmatch(filename, pattern):
                logger.info(f"skipping {filename} down't match {pattern} ")
                continue
            df = pd.read_csv(StringIO(zf.read(filename).decode(encoding='utf-8')), **read_csv_opts)
            data[filename] = df
    return data


def ts_store(
    data: pd.DataFrame,
    *,
    extension: Literal['csv', 'tsv', 'gephi', 'txt', 'json', 'xlsx', 'clipboard'],
    basename: str,
    sep: str = '\t',
):
    filename = f"{now_timestamp()}_{basename}.{extension}"

    if extension == 'xlsx':
        data.to_excel(filename)
    elif extension in ('csv', 'tsv', 'gephi', 'txt'):
        data.to_csv(filename, sep=sep)
    elif extension in ('json'):
        data.to_json(filename)
    elif extension == 'clipboard':
        data.to_clipboard(sep=sep)
        filename = "clipboard"
    else:
        raise ValueError(f"unknown extension: {extension}")
    logger.info(f'Data stored in {filename}')


def rename_columns(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    df.columns = columns
    return df


def as_slim_types(df: pd.DataFrame, columns: list[str], dtype: np.dtype) -> pd.DataFrame:
    if df is None:
        return None
    if isinstance(columns, str):
        columns = [columns]
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna(0).astype(dtype)
    return df


def set_index(df: pd.DataFrame, columns: str | list[str], drop: bool = True, axis_name: str = None) -> pd.DataFrame:
    """Set index if columns exist, otherwise skip (assuming columns already are index)"""
    columns: list[str] = [columns] if isinstance(columns, str) else columns
    if any(column not in df.columns for column in columns):
        return df
    df = df.set_index(columns, drop=drop)
    if axis_name:
        df = df.rename_axis(axis_name)
    return df
