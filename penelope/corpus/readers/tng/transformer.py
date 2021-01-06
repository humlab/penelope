from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Callable, List, TypeVar, Union

import ftfy
import textacy.preprocessing as preprocessing


@dataclass
class ITextTransform:
    transform: Callable = None
    name: str = None
    enabled: bool = None


class TEXT_TRANSFORMS:
    fix_hyphenation = preprocessing.normalize_hyphenated_words
    fix_unicode = preprocessing.normalize_unicode
    fix_whitespaces = preprocessing.normalize_whitespace
    fix_accents = preprocessing.remove_accents
    fix_currency_symbols = preprocessing.replace_currency_symbols
    fix_ftfy_text = ftfy.fix_text


@unique
class KnownTransformType(IntEnum):
    fix_hyphenation = 1
    fix_unicode = 2
    fix_whitespaces = 3
    fix_accents = 4
    fix_currency_symbols = 5
    fix_ftfy_text = 6

    @property
    def transform(self):
        return getattr(TEXT_TRANSFORMS, self.name)


TransformTypeArg = Union[KnownTransformType, List[KnownTransformType], str, List[str], Callable[[str], str]]


class TextTransformOpts:
    def __init__(self, transforms: List[TransformTypeArg] = None):
        self.opts = []
        self.add(
            transforms
            if transforms is not None
            else [KnownTransformType.fix_whitespaces, KnownTransformType.fix_hyphenation]
        )

    def as_known_types(self, ys: TransformTypeArg) -> List[KnownTransformType]:
        ys = ys if isinstance(ys, (list, tuple, )) else [ys]
        return [(KnownTransformType[y] if isinstance(y, str) else y) for y in ys ]

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

    def __add__(self, ys: TransformTypeArg):
        self.add(ys)

    def __sub__(self, ys: TransformTypeArg):
        self.remove(ys)

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


T = TypeVar("T", str, List[str])


class Transformer:
    """Transforms applied on tokenized text"""

    def __init__(self):
        self.transforms = []

    def add(self, transform: Callable[[T], str], add_only_if: bool = True) -> Transformer:
        if add_only_if:
            self.transforms.append(transform)
        return self

    def transform(self, data: T) -> Transformer:

        for ft in self.transforms:
            data = [x for x in ft(data)]

        return data


class TextTransformer(Transformer):
    """Transforms applied on non-tokenized text"""

    def __init__(self, *, text_transform_opts: TextTransformOpts = None):
        super().__init__()
        self.ingest(text_transform_opts)

    def ingest(self, opts: TextTransformOpts) -> TextTransformer:
        for t in opts or []:
            self.add(t.transform)
        return self

    def transform(self, data: T) -> Transformer:

        if isinstance(data, list):
            raise ValueError("text transforms cannot be applied on tokenized data!")

        for ft in self.transforms:
            data = [x for x in ft(data)]

        return data