from __future__ import annotations

from typing import Any, Callable, TypeVar

from penelope.utility.utils import CommaStr

from . import transforms as tr

TransformType = str | list[str] | Callable[[str], str]

T = TypeVar("T", str, list[str])


# pylint: disable=too-many-return-statements
class TransformProperty:
    """Descriptor for transform options. Modifies the transform property of the owner class"""

    def __init__(self, alias: str = None, negate: bool = False):
        self.alias: str = alias.replace('_', '-') if alias else None
        self.negate: bool = negate

    def __set_name__(self, owner: "TransformOpts", name: str):
        self.name: str = name.replace('_', '-')  # pylint: disable=attribute-defined-outside-init

    def __get__(self, obj: "TransformOpts", objtype: str = None) -> Any:  # pylint: disable=unused-argument
        name: str = self.alias or self.name
        for key in obj.find_keys(name):
            if key == name:
                return not self.negate
            if key.startswith(f"{name}?"):
                value: str = key.split('?')[1]
                if value.isdigit():
                    """E.g. 'min-chars?2,...'"""
                    return int(value)
                if value in ('True', 'False'):
                    """E.g. 'remove-symbols?False,...'"""
                    return value == 'True'
                if value in ('None',):
                    return None
                if value == '':
                    """Empty string is interpreted as True e.g. 'remove-symbols?,...'"""
                    return True
                return value
        return self.negate

    def __set__(self, obj: "TransformOpts", value: Any):
        name: str = self.alias or self.name
        if isinstance(value, bool):
            if self.negate:
                value = not value
            if value:
                obj.add(name)
            else:
                obj.remove(name)
        else:
            if value is None:
                obj.remove(name)
            else:
                obj.add(f"{name}?{value}")


class TransformOpts:
    registry: tr.TransformRegistry = None
    DEFAULT_TRANSFORMS: str = CommaStr("")

    def __init__(
        self,
        transforms: str | CommaStr | dict[str, bool] = None,
        extras: list[tr.Transform] = None,
    ):
        self.transforms: CommaStr = CommaStr("")
        self.ingest(self.DEFAULT_TRANSFORMS if transforms is None else transforms)
        self.extras: list[tr.Transform] = extras or []

    def ingest(self, transforms: str | CommaStr | dict[str, bool]) -> TransformOpts:
        if not transforms:
            return self
        if isinstance(transforms, CommaStr):
            self.add(transforms)
        elif isinstance(transforms, str):
            self.add(CommaStr(transforms))

        if isinstance(transforms, dict):
            for key, value in transforms.items():
                if not self.registry.is_registered(key):
                    raise ValueError(f"Invalid transform: {key}")
                if isinstance(value, bool) or value is None:
                    if not value:
                        self.remove(key)
                    elif value:
                        self.add(key)
                else:
                    self.add(f"{key}?{value}")
        return self

    def add(self, *ys: TransformType) -> TransformOpts:
        for y in ys:
            if callable(y):
                self.extras.append(y)
            elif isinstance(y, str):
                self.transforms += y.replace('_', '-')
            else:
                raise ValueError(f"Invalid transform: {y}")
        return self

    def remove(self, *keys: TransformType) -> TransformOpts:
        for key in keys:
            if callable(key) and key in self.extras:
                self.extras.remove(key)
            elif isinstance(key, str):
                for k in self.find_keys(key):
                    self.transforms -= k
        return self

    def find_keys(self, key: str) -> list[str]:
        key = key.replace('_', '-')
        if '?' in key:
            key: str = key.split('?')[0]
        for part in self.transforms.parts():
            if part == key or part.startswith(f"{key}?"):
                yield part

    def clear(self: TransformType) -> TransformOpts:
        self.transforms = ""
        self.extras = []
        return self

    def __add__(self, ys: TransformType) -> TransformOpts:
        self.add(ys)
        return self

    def __sub__(self, ys: TransformType) -> TransformOpts:
        self.remove(ys)
        return self

    def __iter__(self):
        return iter(self.transforms)

    @staticmethod
    def empty():
        return TransformOpts().clear()

    @property
    def props(self) -> list[str]:
        return {x.split('?')[0]: x.split('?')[1] if '?' in x else True for x in self.transforms.parts()}

    def transform(self, data: T) -> T:
        if self.no_effect():
            return data
        return self.getfx()(data)

    def getfx(self) -> tr.Transform:
        if self.no_effect():
            return lambda x: x
        return self.registry.getfx(self.transforms, extras=self.extras)

    def no_effect(self):
        return not self.transforms and not self.extras


class TextTransformOpts(TransformOpts):
    registry: tr.TransformRegistry = tr.TextTransformRegistry
    DEFAULT_TRANSFORMS = CommaStr('dehyphen,normalize-whitespace')


class TextTransformer:
    """FIXME: Deprecate!"""

    """Transforms applied on non-tokenized text"""

    def __init__(self, *, transform_opts: TextTransformOpts = None):
        self.transform_opts = transform_opts or TextTransformOpts()

    def transform(self, data: T) -> T:
        return self.transform_opts.transform(data)
