from __future__ import annotations

import functools
import inspect
import io
from dataclasses import dataclass, field, fields
from inspect import isclass
from pathlib import Path
from typing import Any, Callable, Generic, Type, TypeVar

import yaml

from penelope.utility.dots import dget, dotexists

T = TypeVar("T", str, int, float)


class SafeLoaderIgnoreUnknown(yaml.SafeLoader):  # pylint: disable=too-many-ancestors
    def let_unknown_through(self, node):  # pylint: disable=unused-argument
        return None


SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.let_unknown_through)


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context: str = "default"

    def get(self, *keys: str, default: Any | Type[Any] = None, mandatory: bool = False) -> Any:
        if mandatory and not self.exists(*keys):
            raise ValueError(f"Missing mandatory key: {keys}")

        value: Any = dget(self, *keys)

        if value is not None:
            return value

        return default() if isclass(default) else default

    def exists(self, *keys) -> bool:
        return dotexists(self, *keys)

    @staticmethod
    def load(*, source: str | dict | Config = None, context: str = None) -> "Config":
        if isinstance(source, Config):
            return source
        data = source
        data: dict = (
            (
                yaml.load(Path(source).read_text(encoding="utf-8"), Loader=SafeLoaderIgnoreUnknown)
                if source.endswith(".yml") or source.endswith(".yaml")
                else yaml.load(io.StringIO(source), Loader=SafeLoaderIgnoreUnknown)
            )
            if isinstance(source, str)
            else source
        )
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, found {type(data)}")
        return Config(data, context=context)


@dataclass
class Configurable:
    def resolve(self):
        for attrib in fields(self):
            if isinstance(getattr(self, attrib.name), ConfigValue):
                setattr(self, attrib.name, getattr(self, attrib.name).resolve())


@dataclass
class ConfigValue(Generic[T]):
    key: str | Type[T]
    default: T | None = None
    description: str | None = None
    after: Callable[[T], T] | None = None
    mandatory: bool = False

    def resolve(self) -> T:
        if isinstance(self.key, Config):
            return ConfigStore.config
        if isclass(self.key):
            return self.key()
        if self.mandatory and not self.default:
            if not ConfigStore.config.exists(self.key):
                raise ValueError(f"ConfigValue {self.key} is mandatory but missing from config")

        value = ConfigStore.config.get(*self.key.split(","), default=self.default)
        if value and self.after:
            return self.after(value)
        return value

    @staticmethod
    def create_field(key: str, default: Any = None, description: str = None) -> Any:
        return field(  # pylint: disable=invalid-field-call
            default_factory=lambda: ConfigValue(key=key, default=default, description=description).resolve()
        )


class ConfigStore:
    store: dict[str, str | Config] = {"default": "./config.yml"}
    context: str = "default"

    @classmethod
    def configure_context(cls, context: str, source: str | dict | Config) -> Config:
        return cls._set_config(
            context=context,
            cfg=source,
        )

    @classmethod
    @property
    def config(cls) -> "Config":
        if isinstance(cls.store.get(cls.context), Config):
            return cls.store[cls.context]
        return cls.load(context=cls.context)

    @classmethod
    def resolve(cls, value: T | ConfigValue) -> T:
        if not isinstance(value, ConfigValue):
            return value
        return dget(cls.config, value.key)

    @classmethod
    def load(cls, *, context: str = "default", source: Config | str | dict = None) -> "Config":
        if isinstance(source, (dict, Config)):
            return cls._set_config(context=context, cfg=source)

        current: str | dict | Config | None = cls.store.get(context)

        if not current and not source:
            raise ValueError(f"Config context {context} not found")

        if isinstance(current, Config):
            return current

        if isinstance(current, dict):
            cls._set_config(context=context, cfg=current)

        return cls._set_config(
            context=context,
            cfg=Config.load(source=source or cls.store.get(context), context=context),
        )

    @classmethod
    def _set_config(cls, *, context: str = "default", cfg: Config | dict = None) -> "Config":
        cfg: Config = cfg if isinstance(cfg, Config) else Config.load(source=cfg, context=context)
        cfg.context = context
        cls.store[context] = cfg
        cls.context = context
        return cls.store[context]


configure_context = ConfigStore.configure_context


def resolve_arguments(fn_or_cls, args, kwargs):
    """Resolve any ConfigValue arguments in a function or class constructor"""
    kwargs = {
        k: v.default
        for k, v in inspect.signature(fn_or_cls).parameters.items()
        if isinstance(v.default, ConfigValue) and v.default is not inspect.Parameter.empty
    } | kwargs
    args = (a.resolve() if isinstance(a, ConfigValue) else a for a in args)
    for k, v in kwargs.items():
        if isinstance(v, ConfigValue):
            kwargs[k] = v.resolve()
    return args, kwargs


def inject_config(fn_or_cls: T) -> Callable[..., T]:
    @functools.wraps(fn_or_cls)
    def decorated(*args, **kwargs):
        args, kwargs = resolve_arguments(fn_or_cls, args, kwargs)
        return fn_or_cls(*args, **kwargs)

    return decorated
