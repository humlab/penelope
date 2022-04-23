from __future__ import annotations

import fnmatch
import operator
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from io import StringIO
from numbers import Number
from operator import methodcaller
from typing import Any, Callable, Dict, List, Literal, Mapping, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from .filename_utils import replace_extension
from .utils import now_timestamp, revdict

DataFrameFilenameTuple = Tuple[pd.DataFrame, str]


def unstack_data(data: pd.DataFrame, pivot_keys: List[str]) -> pd.DataFrame:
    """Unstacks a dataframe that has been grouped by temporal_key and pivot_keys"""
    if len(pivot_keys) <= 1 or data is None:
        return data
    data: pd.DataFrame = data.set_index(pivot_keys)
    while isinstance(data.index, pd.MultiIndex):
        data = data.unstack(level=1, fill_value=0)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [' '.join(x) for x in data.columns]
    return data


def faster_to_dict_records(df: pd.DataFrame) -> List[dict]:
    data: List[Any] = df.values.tolist()
    columns: List[str] = df.columns.tolist()
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
        Tuple length must be 2 or 3 and first element must be sign, second (optional) a binary op.
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
        criterias (dict): Dict with masking criterias

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
        attr_operator: Union[str, Callable] = None

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

    def hot_attributes(self, doc: pd.DataFrame) -> List[str]:
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


PivotKeySpec = Mapping[str, Union[str, Mapping[str, int]]]


@dataclass
class PivotKeys:
    """Simple helper for pre-defined pivot keys where each keys has a given value-name/id-name mapping.

    Args:

        pivot_key_specs (Mapping[dict]): Specifies avaliable pivot keys, mapping names-to-ids and value ranges

        Sample in-data format:
        {
            'grönsak': {
                'text_name': 'grönsak',     # Pivot key text (column) name, or presentable name
                'id_name': 'grönsak_id',    # # Pivot key ID column name
                'values': {'unknown': 0, 'gurka': 1, 'tomat': 2}
            },
            ...
        }

    """

    pivot_keys: Mapping[str, PivotKeySpec]

    def __post_init__(self):
        """Changes mapping to a dict of dicts instead of a list of dicts"""
        self.pivot_keys = self.pivot_keys or []
        if isinstance(self.pivot_keys, list):
            self.pivot_keys = {x['text_name']: x for x in self.pivot_keys} if self.pivot_keys else {}
        self.is_satisfied()

    def pivot_key(self, text_name: str) -> dict:
        return self.pivot_keys.get(text_name, {})

    def __getitem__(self, text_name: str) -> dict:
        return self.pivot_key(text_name)

    def __len__(self):
        return len(self.pivot_keys)

    @cached_property
    def key_name2key_id(self) -> dict:
        return {x['text_name']: x['id_name'] for x in self.pivot_keys.values()}

    @cached_property
    def key_id2key_name(self) -> dict:
        """Translates e.g. `gender_id` to `gender`."""
        return revdict(self.key_name2key_id)

    @property
    def text_names(self) -> List[str]:
        return [x for x in self.pivot_keys]

    @cached_property
    def id_names(self) -> List[str]:
        return [x.get('id_name') for x in self.pivot_keys.values()]

    @property
    def has_pivot_keys(self) -> List[str]:
        return len(self.text_names) > 0

    def key_value_name2id(self, text_name: str) -> Mapping[str, int]:
        """Returns name/id mapping for given key's value range"""
        return self.pivot_key(text_name)['values']

    def key_value_id2name(self, text_name: str) -> Mapping[int, str]:
        """Returns id/name mapping for given key's value range"""
        return revdict(self.key_value_name2id(text_name))

    def key_values_str(self, names: Set[str], sep=': ') -> List[str]:
        return [f'{k}{sep}{v}' for k in names for v in self.key_value_name2id(k).keys()]

    def is_satisfied(self) -> bool:

        if self.pivot_keys is None:
            return True

        if not isinstance(self.pivot_keys, (list, dict)):
            raise TypeError(f"expected list/dict of pivot key specs, got {type(self.pivot_keys)}")

        items: dict = self.pivot_keys if isinstance(self.pivot_keys, list) else self.pivot_keys.values()

        if not all(isinstance(x, dict) for x in items):
            raise TypeError("expected list of dicts")

        expected_keys: Set[str] = {'text_name', 'id_name', 'values'}
        if len(items) > 0:
            if not all(set(x.keys()) == expected_keys for x in items):
                raise TypeError("expected list of dicts(id_name,text_name,values)")

        return True

    def is_id_name(self, name: str) -> bool:
        return any(v['id_name'] == name for _, v in self.pivot_keys.items())

    def is_text_name(self, name: str) -> bool:
        return any(k == name for k in self.pivot_keys)

    def create_filter_by_value_pairs(
        self, value_pairs: List[str], sep: str = ': ', vsep: str = ','
    ) -> PropertyValueMaskingOpts:
        """Create a filter from list of [ 'key1=v1', 'key1=v2' 'k3=v5', ....]   (sep '=' is an argument)"""

        """Convert list of pairs to dict of list: {'key1: [v1, v2], 'k3': [v5]...}"""
        key_values = defaultdict(list)
        value_tuples: Tuple[str, str] = [x.split(sep) for x in value_pairs]
        for k, v in value_tuples:
            is_sequence_of_values: bool = vsep is not None and vsep in v
            values: List[str | int] = v.split(vsep) if is_sequence_of_values else [v]
            try:
                values = [int(x) for x in values]
            except TypeError:
                ...
            key_values[k].extend(values)

        opts = self.create_filter_key_values_dict(key_values, decode=True)

        return opts

    def create_filter_key_values_dict(
        self, key_values: Mapping[str, List[str | int]], decode: bool = True
    ) -> PropertyValueMaskingOpts:
        """Create a filter from dict of list: {'key1: [v1, v2], 'k3': [v5]...}"""
        opts = PropertyValueMaskingOpts()
        if not decode:
            """Values are e.g. ('xxx_id', [1,2,3,...}"""
            for k, v in key_values.items():
                opts[k] = list(map(int, v or [])) if self.is_id_name(k) else list(v)

        else:
            """Values are e.g. ('xxx', ['label-1','label2','label-3',...}"""
            for k, v in key_values.items():
                fg: Callable[[str], int] = self.key_value_name2id(k).get
                opts[self.key_name2key_id[k]] = [int(fg(x)) for x in v]
        return opts

    def create_filter_by_str_sequence(
        self,
        key_value_pairs: List[str],
        decode: bool = True,
        sep: str = ': ',
        vsep: str = None,
    ) -> PropertyValueMaskingOpts:
        """Returns user's filter selections as a name-to-values mapping.

        key_value_pairs ::= { key_value_pair }
        key_value_pair  ::= key "sep" value
        key             ::= "str"
        value           ::= "int" [ "vsep" value ]
                          | "str" [ "vsep" value ]
        Args:
            key_value_pairs ([List[str]], optional):  Sequence of key-values.
            decode (bool, optional): decode text name/value to id name/values. Defaults to True.
            sep (str, optional): Key-value delimiter. Defaults to '='.
            vsep (str, optional): Value list delimiter. Defaults to None.

        Returns:
            PropertyValueMaskingOpts: [description]
        """
        key_values_dict = defaultdict(list)
        for k, v in map(methodcaller("split", sep), key_value_pairs):
            if vsep is not None and vsep in v:
                v = v.split(vsep)
                key_values_dict[k].extend(v)
            else:
                key_values_dict[k].append(v)
        filter_opts = self.create_filter_key_values_dict(key_values_dict, decode=decode)
        return filter_opts

    def decode_pivot_keys(self, df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        """Decode pivot key id-columns in `df` and add a new text-column.
        All columns in `df` with a name found in `id_names` are decoded.
        Found id-columns are dropped if `drop` is true.
        """
        for key_id in self.id_names:
            if key_id in df.columns:
                key_name: str = self.key_id2key_name.get(key_id)
                id2name: Callable[[int], str] = self.key_value_id2name(text_name=key_name).get
                df[key_name] = df[key_id].apply(id2name)
                if drop:
                    df.drop(columns=key_id, inplace=True, errors='ignore')
        return df


def try_split_column(
    df: pd.DataFrame,
    source_name: str,
    sep: str,
    target_names: List[str],
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
    zip_filename: str, dfs: Union[DataFrameFilenameTuple, List[DataFrameFilenameTuple]], extension='csv', **to_csv_opts
):
    if not isinstance(dfs, (list, tuple)):
        raise ValueError("expected tuple or list of tuples")

    if isinstance(dfs, (tuple,)):
        dfs = [dfs]

    with zipfile.ZipFile(zip_filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for (df, filename) in dfs:
            if not isinstance(df, pd.core.frame.DataFrame) or not isinstance(filename, str):
                raise ValueError(
                    f"Expected Tuple[pd.DateFrame, filename: str], found Tuple[{type(df)}, {type(filename)}]"
                )
            filename = replace_extension(filename=filename, extension=extension)
            data_str = df.to_csv(**to_csv_opts)
            zf.writestr(filename, data=data_str)


def pandas_read_csv_zip(zip_filename: str, pattern='*.csv', **read_csv_opts) -> Dict:

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


def rename_columns(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    df.columns = columns
    return df


def as_slim_types(df: pd.DataFrame, columns: List[str], dtype: np.dtype) -> pd.DataFrame:
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
