import glob
from collections import defaultdict
from operator import methodcaller
from os.path import isdir, isfile, join
from typing import Any, Callable, Self

import pandas as pd
from loguru import logger

from .file_utility import read_yaml
from .pandas_utils import PropertyValueMaskingOpts
from .utils import dotcoalesce, revdict

PivotKeySpec = dict[str, str | dict[str, int]]

# pylint: disable=too-many-public-methods


class PivotKeys:
    """Simple helper for pre-defined pivot keys where each keys has a given value-name/id-name mapping.

    Args:

        pivot_key_specs (dict[dict]): Specifies avaliable pivot keys, mapping names-to-ids and value ranges

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

    def __init__(self, pivot_keys: dict[str, PivotKeySpec] | list[PivotKeySpec] = None):
        self._pivot_keys_spec: dict[str, PivotKeySpec] = {}
        self.update(pivot_keys)

    def update(self, other: Self | dict | list | str) -> Self:
        if isinstance(other, PivotKeys):
            self._pivot_keys_spec.update(other.pivot_keys_spec)
        if isinstance(other, dict):
            self._pivot_keys_spec.update(other)
        if isinstance(other, list):
            self._pivot_keys_spec.update({x['text_name']: x for x in other})
        if isinstance(other, str):
            if isfile(other):
                data: dict = PivotKeys.try_load(other, default={})
                if data is not None:
                    return PivotKeys(pivot_keys=data)
            if isdir(other):
                return PivotKeys.load_by_probe(other)
            logger.warning(f"Pivot keys file not found: {other}")
            return PivotKeys()
        self.is_satisfied()
        return self

    @staticmethod
    def create_by_index(document_index: pd.DataFrame, *text_columns: str) -> Self:
        """Create pivot keys from document index. For each text column an ID column is converted, and used as a pivot key."""
        # FIXME: Extend to support existing ID columns, naming, etc
        pivot_keys: dict = {}
        for text_column in text_columns:
            id_column: str = f"{text_column}_id"
            pivot_keys[text_column] = {
                'text_name': text_column,
                'id_name': id_column,
                'values': codify_column(document_index, text_column, id_column_name=id_column),
            }
        return PivotKeys(pivot_keys=pivot_keys)

    @staticmethod
    def try_load(path: str, default: dict = None) -> Self:
        data: dict = read_yaml(path)
        return dotcoalesce(data, 'extra_opts.pivot_keys', 'pivot_keys', default=default)

    @staticmethod
    def load_by_probe(folder: str) -> Self:
        """Probes folder for pivot keys"""

        if isfile(join(folder, 'pivot_keys.yml')):
            return PivotKeys(join(folder, 'pivot_keys.yml'))

        for path in glob.glob(folder + '/*.y*ml'):
            data: dict = PivotKeys.try_load(path, default=None)
            if data is not None:
                return PivotKeys(pivot_keys=data)

        return PivotKeys()

    @property
    def pivot_keys_spec(self) -> dict[str, PivotKeySpec]:
        return self._pivot_keys_spec

    @pivot_keys_spec.setter
    def pivot_keys_spec(self, pivot_keys: dict[str, PivotKeySpec] | list[PivotKeySpec]) -> None:
        self += pivot_keys or {}

    def __contains__(self, text_name: str) -> bool:
        return text_name in self._pivot_keys_spec

    def __getitem__(self, text_name: str, default: dict = None) -> dict:
        return self._pivot_keys_spec.get(text_name, default or {})

    def get(self, text_name: str, default: Any = None) -> dict:
        return self._pivot_keys_spec.get(text_name, default or {})

    def __len__(self) -> int:
        return len(self._pivot_keys_spec)

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        if isinstance(other, dict):
            return self._pivot_keys_spec == other
        if isinstance(other, PivotKeys):
            """FIXME: Might need to do a deep comparison"""
            return self._pivot_keys_spec == other._pivot_keys_spec
        return False

    def __add__(self, other) -> Self:
        self.update(other)
        return self

    @property
    def key_name2key_id(self) -> dict:
        return {x['text_name']: x['id_name'] for x in self._pivot_keys_spec.values()}

    @property
    def key_id2key_name(self) -> dict:
        """Translates e.g. `gender_id` to `gender`."""
        return revdict(self.key_name2key_id)

    @property
    def text_names(self) -> list[str]:
        return [x['text_name'] for x in self._pivot_keys_spec.values()]

    @property
    def id_names(self) -> list[str]:
        return [x['id_name'] for x in self._pivot_keys_spec.values()]

    @property
    def has_pivot_keys(self) -> list[str]:
        return len(self._pivot_keys_spec) > 0

    def key_value_name2id(self, text_name: str) -> dict[str, int]:
        """Returns name/id mapping for given key's value range"""
        return self.get(text_name).get('values')

    def key_value_id2name(self, text_name: str) -> dict[int, str]:
        """Returns id/name mapping for given key's value range"""
        return revdict(self.key_value_name2id(text_name))

    def key_values_str(self, names: set[str], sep=': ') -> list[str]:
        return [f'{k}{sep}{v}' for k in names for v in self.key_value_name2id(k).keys()]

    def is_satisfied(self) -> bool:
        if self._pivot_keys_spec is None:
            return True

        if not isinstance(self._pivot_keys_spec, (list, dict)):
            raise TypeError(f"expected list/dict of pivot key specs, got {type(self._pivot_keys_spec)}")

        items: dict = (
            self._pivot_keys_spec if isinstance(self._pivot_keys_spec, list) else self._pivot_keys_spec.values()
        )

        if not all(isinstance(x, dict) for x in items):
            raise TypeError("expected list of dicts")

        expected_keys: set[str] = {'text_name', 'id_name', 'values'}
        if len(items) > 0:
            if not all(set(x.keys()) == expected_keys for x in items):
                raise TypeError("expected list of dicts(id_name,text_name,values)")

        return True

    def is_id_name(self, name: str) -> bool:
        return any(v['id_name'] == name for _, v in self.pivot_keys_spec.items())

    def is_text_name(self, name: str) -> bool:
        return any(k == name for k in self.pivot_keys_spec)

    def create_filter_by_value_pairs(
        self, value_pairs: list[str], sep: str = ': ', vsep: str = ','
    ) -> PropertyValueMaskingOpts:
        """Create a filter from list of [ 'key1=v1', 'key1=v2' 'k3=v5', ....]   (sep '=' is an argument)"""

        """Convert list of pairs to dict of list: {'key1: [v1, v2], 'k3': [v5]...}"""
        key_values = defaultdict(list)
        value_tuples: tuple[str, str] = [x.split(sep) for x in value_pairs]
        for k, v in value_tuples:
            is_sequence_of_values: bool = vsep is not None and vsep in v
            values: list[str | int] = v.split(vsep) if is_sequence_of_values else [v]
            try:
                values = [int(x) for x in values]
            except TypeError:
                ...
            key_values[k].extend(values)

        opts = self.create_filter_key_values_dict(key_values, decode=True)

        return opts

    @property
    def single_key_options(self) -> dict:
        return {v['text_name']: v['id_name'] for v in self.pivot_keys_spec.values()}

    def create_filter_key_values_dict(
        self, key_values: dict[str, list[str | int]], decode: bool = True
    ) -> PropertyValueMaskingOpts:
        """Create a filter from dict of list: {'key1: [v1, v2], 'k3': [v5]...}"""
        opts = PropertyValueMaskingOpts()
        if not decode:
            """Values are e.g. ('xxx_id', [1,2,3,...}"""
            for k, v in key_values.items():
                opts[k] = list(map(int, v or [])) if self.is_id_name(k) else list(v)

        else:
            """Values are e.g. ('xxx', ['label_1','label_2','label_3',...}"""
            key2id: dict = self.key_name2key_id
            for k, v in key_values.items():
                fg: Callable[[str], int] = self.key_value_name2id(k).get
                opts[key2id[k]] = [int(fg(x)) for x in v]
        return opts

    def create_filter_by_str_sequence(
        self,
        key_value_pairs: list[str],
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
            key_value_pairs ([list[str]], optional):  Sequence of key-values.
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


def codify_column(document_index: pd.DataFrame, column_name: str, id_column_name: str = None) -> dict[Any, int]:
    """Create a new column named id_column_name with unique integer values for each unique value in column_name"""

    if column_name not in document_index.columns:
        raise ValueError(f'Column {column_name} not found in document_index')

    if id_column_name is None:
        id_column_name = f'{column_name}_id'

    if id_column_name in document_index.columns:
        raise ValueError(f'Column {id_column_name} already exists in document_index')

    categorical: pd.Categorical = pd.Categorical(document_index[column_name])

    document_index[id_column_name] = categorical.codes

    category_to_id: dict[Any, int] = {category: id for id, category in enumerate(categorical.categories)}

    return category_to_id
