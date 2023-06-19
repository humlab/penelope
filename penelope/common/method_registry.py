from __future__ import annotations

import os
from typing import Any, Callable, Type, TypeVar

import pandas as pd
import requests


def revdict(d: dict) -> dict:
    return {v: k for k, v in d.items()}


def group_to_list_of_records2(df: pd.DataFrame, key: str) -> dict[str | int, list[dict]]:
    """Groups `df` by `key` and aggregates each group to list of row records (dicts)"""
    return {q: df.loc[ds].to_dict(orient='records') for q, ds in df.groupby(key).groups.items()}


def group_to_list_of_records(
    df: pd.DataFrame, key: str, properties: list[str] = None, ctor: Type = None
) -> dict[str | int, list[dict]]:
    """Groups `df` by `key` and aggregates each group to list of row records (dicts)"""
    key_rows: pd.DataFrame = pd.DataFrame(
        data={
            key: df[key],
            'data': (df[properties] if properties else df).to_dict("records"),
        }
    )
    if ctor is not None:
        key_rows['data'] = key_rows['data'].apply(lambda x: ctor(**x))

    return key_rows.groupby(key)['data'].apply(list).to_dict()


def download_url_to_file(url: str, target_name: str, force: bool = False) -> None:
    if os.path.isfile(target_name):
        if not force:
            raise ValueError("File exists, use `force=True` to overwrite")
        os.unlink(target_name)

    ensure_path(target_name)

    with open(target_name, 'w', encoding="utf-8") as fp:
        data: str = requests.get(url, allow_redirects=True, timeout=600).content.decode("utf-8")
        fp.write(data)


def probe_filename(filename: list[str], exts: list[str] = None) -> str | None:
    """Probes existence of filename with any of given extensions in folder"""
    for probe_name in set([filename] + ([replace_extension(filename, ext) for ext in exts] if exts else [])):
        if os.path.isfile(probe_name):
            return probe_name
    raise FileNotFoundError(filename)


def replace_extension(filename: str, extension: str) -> str:
    if filename.endswith(extension):
        return filename
    base, _ = os.path.splitext(filename)
    return f"{base}{'' if extension.startswith('.') else '.'}{extension}"


def ensure_path(f: str) -> None:
    os.makedirs(os.path.dirname(f), exist_ok=True)


S = TypeVar("S")
T = TypeVar("T")
Method = Callable[[S], T]


class MethodRegistry:
    _items: dict[str, Any] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def items(cls) -> dict[str, Any]:
        return cls._items

    @classmethod
    def add(cls, fn: Method, key: str = None) -> Method:
        """Add transform function"""
        if key is None:
            key = fn.__name__
        keys = [k.strip() for k in key.replace('_', '-').split(',')]
        cls._items[keys[0]] = fn
        if len(keys) > 1:
            for k in keys[1:]:
                if k in cls._items:
                    continue
                cls._aliases[k] = keys[0]
        return fn

    @classmethod
    def get(cls, key: str) -> Method:
        """Get transform function by key"""
        key = key.replace('_', '-').strip()

        if key in cls._items:
            return cls._items.get(key)

        if key in cls._aliases:
            return cls._items.get(cls._aliases[key])

        if '?' in key:
            """Method has arguments"""
            return cls.add(fn=cls.partial_to_total(key), key=key)

        raise ValueError(f"preprocessor {key} is not registered")

    @classmethod
    def partial_to_total(cls, key: str) -> Method:
        """Method functions needs or accepts extra with arguments
        Examples:
            'min-chars?chars=2' => kwargs = {'chars': '2'}
            'min-chars?2'       => args = '2'
            'any-option?[2,3]'  => args = '[2,3]'

        """
        key = key.replace('_', '-')
        pkey, args = key.split('?')

        if '[' in args:
            """args is a list"""
            args = [x.strip() for x in args.lstrip('[').rstrip(']').split(',')]
            return lambda x: cls.get(pkey)(x, *args)  # pylint: disable=unnecessary-lambda

        if '=' in args:
            kwargs = {k: v for k, v in [x.split('=') for x in args.split('&')]}
            return lambda x: cls.get(pkey)(x, **kwargs)  # pylint: disable=unnecessary-lambda

        return lambda x: cls.get(pkey)(x, args)

    @classmethod
    def gets(cls, *keys: tuple[str]) -> list[Method]:
        return [cls.get(k) for key in keys for k in key.split(',')]

    # @classmethod
    # def getfx(cls, *keys: tuple[str], extras: list = None) -> Method:
    #     fxs: list[Callable[..., Any]] = [cls.get(k) for key in keys for k in key.split(',') if k]
    #     if extras:
    #         fxs.extend(extras)
    #     if not fxs:
    #         return lambda x: x
    #     return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(fxs))

    @classmethod
    def register(cls, key: str = None, kind: str = None, **args):
        def decorator(fn):
            if kind == "function":
                fn = fn(**args)
            cls.add(fn, key)
            return fn

        return decorator

    @classmethod
    def is_registered(cls, key: str):
        try:
            return cls.get(key) is not None
        except ValueError:
            return False

    @classmethod
    def __getitem__(cls, key: str) -> Method:
        return cls.get(key)

    @classmethod
    def __contains__(cls, key: str) -> bool:
        return cls.is_registered(key)
