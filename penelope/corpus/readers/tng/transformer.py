from __future__ import annotations

from typing import Callable, List, TypeVar, Union

from ...transforms import KnownTransformType

TransformTypeArg = Union[KnownTransformType, List[KnownTransformType], str, List[str], Callable[[str], str]]

T = TypeVar("T", str, List[str])


class TextTransformOpts:
    def __init__(self, transforms: List[TransformTypeArg] = None):
        self.opts = []
        self.add(
            transforms
            if transforms is not None
            else [KnownTransformType.fix_whitespaces, KnownTransformType.fix_hyphenation]
        )

    def as_known_types(self, ys: TransformTypeArg) -> List[KnownTransformType]:
        ys = (
            ys
            if isinstance(
                ys,
                (
                    list,
                    tuple,
                ),
            )
            else [ys]
        )
        return [(KnownTransformType[y] if isinstance(y, str) else y) for y in ys]

    def add(self, ys: TransformTypeArg) -> TextTransformOpts:
        for y in self.as_known_types(ys):
            if y not in self.opts:
                self.opts.append(y)
        return self

    def remove(self, ys: TransformTypeArg) -> TextTransformOpts:
        self.opts = [x for x in self.opts if x not in self.as_known_types(ys)]
        return self

    def clear(self: TransformTypeArg) -> TextTransformOpts:
        self.opts.clear()
        return self

    def __add__(self, ys: TransformTypeArg) -> TextTransformOpts:
        self.add(ys)
        return self

    def __sub__(self, ys: TransformTypeArg) -> TextTransformOpts:
        self.remove(ys)
        return self

    def __iter__(self):
        return iter(self.opts)

    @staticmethod
    def empty():
        return TextTransformOpts().clear()

    @property
    def props(self) -> List[str]:
        return {x.name: True for x in self.opts}

    def __setattr__(self, key, value):
        if key in KnownTransformType.__members__:
            if value:
                self.add(key)
            else:
                self.remove(key)
        else:
            object.__setattr__(self, key, value)


class TextTransformer:
    """Transforms applied on non-tokenized text"""

    def __init__(self, *, transform_opts: TextTransformOpts = None):
        self.transform_opts = transform_opts or TextTransformOpts()

    def transform(self, data: T) -> T:

        if isinstance(data, list):
            raise ValueError("text transforms cannot be applied on tokenized data!")

        for ft in self.transform_opts.opts:
            data = ft.transform(data)

        return data
