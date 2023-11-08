import glob
from collections import defaultdict
from functools import cached_property
from operator import methodcaller
from os.path import isdir, isfile, join
from typing import Callable

import pandas as pd
from loguru import logger

from .file_utility import read_yaml
from .pandas_utils import PropertyValueMaskingOpts
from .utils import clear_cached_properties, dotcoalesce, revdict

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
        self.pivot_keys: dict[str, PivotKeySpec] = pivot_keys or {}

    @staticmethod
    def load(path: str, default: dict = None) -> "PivotKeys":
        """Loads pivot keys from any given file, or probe in folder for pivot keys"""
        try:
            if isinstance(path, dict):
                return PivotKeys(pivot_keys=path)

            if isinstance(path, str):
                if isfile(path):
                    data: dict = PivotKeys.try_load(path, default=default)
                    if data is not None:
                        return PivotKeys(pivot_keys=data)

                if isdir(path):
                    return PivotKeys.load_by_probe(path)

        except FileNotFoundError:
            ...
        logger.warning(f"Pivot keys file not found: {path}")
        return PivotKeys()

    @staticmethod
    def try_load(path: str, default: dict = None) -> "PivotKeys":
        data: dict = read_yaml(path)
        return dotcoalesce(data, 'extra_opts.pivot_keys', 'pivot_keys', default=default)

    @staticmethod
    def load_by_probe(folder: str) -> "PivotKeys":
        """Probes folder for pivot keys"""

        if isfile(join(folder, 'pivot_keys.yml')):
            return PivotKeys.load(join(folder, 'pivot_keys.yml'))

        for path in glob.glob(folder + '/*.y*ml'):
            data: dict = PivotKeys.try_load(path, default=None)
            if data is not None:
                return PivotKeys(pivot_keys=data)

        return PivotKeys()

    @property
    def pivot_keys(self) -> dict[str, PivotKeySpec]:
        return self._pivot_keys

    @pivot_keys.setter
    def pivot_keys(self, pivot_keys: dict[str, PivotKeySpec] | list[PivotKeySpec]):
        clear_cached_properties(self)
        self._pivot_keys: dict[str, PivotKeySpec] = pivot_keys or {}
        if isinstance(self.pivot_keys, list):
            """Changes mapping to a dict of dicts instead of a list of dicts"""
            self._pivot_keys = {x['text_name']: x for x in self.pivot_keys} if self.pivot_keys else {}
        self.is_satisfied()

    def get(self, text_name: str) -> dict:
        return self.pivot_keys.get(text_name, {})

    def __getitem__(self, text_name: str) -> dict:
        return self.get(text_name)

    def __len__(self):
        return len(self.pivot_keys)

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, dict):
            return self._pivot_keys == other
        if isinstance(other, PivotKeys):
            """FIXME: Might need to do a deep comparison"""
            return self._pivot_keys == other._pivot_keys
        return False

    @cached_property
    def key_name2key_id(self) -> dict:
        return {x['text_name']: x['id_name'] for x in self.pivot_keys.values()}

    @cached_property
    def key_id2key_name(self) -> dict:
        """Translates e.g. `gender_id` to `gender`."""
        return revdict(self.key_name2key_id)

    @property
    def text_names(self) -> list[str]:
        return [x for x in self.pivot_keys]

    @cached_property
    def id_names(self) -> list[str]:
        return [x.get('id_name') for x in self.pivot_keys.values()]

    @property
    def has_pivot_keys(self) -> list[str]:
        return len(self.text_names) > 0

    def key_value_name2id(self, text_name: str) -> dict[str, int]:
        """Returns name/id mapping for given key's value range"""
        return self.get(text_name).get('values')

    def key_value_id2name(self, text_name: str) -> dict[int, str]:
        """Returns id/name mapping for given key's value range"""
        return revdict(self.key_value_name2id(text_name))

    def key_values_str(self, names: set[str], sep=': ') -> list[str]:
        return [f'{k}{sep}{v}' for k in names for v in self.key_value_name2id(k).keys()]

    def is_satisfied(self) -> bool:
        if self.pivot_keys is None:
            return True

        if not isinstance(self.pivot_keys, (list, dict)):
            raise TypeError(f"expected list/dict of pivot key specs, got {type(self.pivot_keys)}")

        items: dict = self.pivot_keys if isinstance(self.pivot_keys, list) else self.pivot_keys.values()

        if not all(isinstance(x, dict) for x in items):
            raise TypeError("expected list of dicts")

        expected_keys: set[str] = {'text_name', 'id_name', 'values'}
        if len(items) > 0:
            if not all(set(x.keys()) == expected_keys for x in items):
                raise TypeError("expected list of dicts(id_name,text_name,values)")

        return True

    def is_id_name(self, name: str) -> bool:
        return any(v['id_name'] == name for _, v in self.pivot_keys.items())

    def is_text_name(self, name: str) -> bool:
        return any(k == name for k in self.pivot_keys)

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

    @cached_property
    def single_key_options(self) -> dict:
        return {v['text_name']: v['id_name'] for v in self.pivot_keys.values()}

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
            """Values are e.g. ('xxx', ['label-1','label2','label-3',...}"""
            for k, v in key_values.items():
                fg: Callable[[str], int] = self.key_value_name2id(k).get
                opts[self.key_name2key_id[k]] = [int(fg(x)) for x in v]
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
