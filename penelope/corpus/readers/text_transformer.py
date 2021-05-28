from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import ftfy
import textacy.preprocessing as preprocessing


class TEXT_TRANSFORMS:
    fix_hyphenation = preprocessing.normalize_hyphenated_words
    fix_unicode = preprocessing.normalize_unicode
    fix_whitespaces = preprocessing.normalize_whitespace
    fix_accents = preprocessing.remove_accents
    fix_currency_symbols = preprocessing.replace_currency_symbols
    fix_ftfy_text = ftfy.fix_text


@dataclass
class TextTransformOpts:

    fix_whitespaces: bool = True
    fix_hyphenation: bool = True
    fix_ftfy_text: bool = True
    fix_accents: bool = False
    fix_unicode: bool = False

    def clear(self):
        self.fix_whitespaces = False
        self.fix_hyphenation = False
        self.fix_ftfy_text = False
        self.fix_accents = False
        self.fix_unicode = False
        return self

    @staticmethod
    def empty():
        return TextTransformOpts().clear()

    extra_transforms: Optional[List[Callable[[str], str]]] = field(default_factory=list)

    @property
    def props(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ('props', 'extra_transforms') and not k.startswith('_') and not callable(v)
        }


class TextTransformer:
    """Transforms applied on non-tokenized text"""

    def __init__(self, *, text_transform_opts: TextTransformOpts = None):
        self.transforms: List[Callable] = []
        self.ingest(text_transform_opts)

    def ingest(self, opts: TextTransformOpts) -> TextTransformer:
        if opts is None:
            return self
        if opts.fix_whitespaces:
            self.fix_whitespaces()
        if opts.fix_hyphenation:
            self.fix_hyphenation()
        if opts.fix_ftfy_text:
            self.fix_ftfy()
        if opts.fix_accents:
            self.fix_accents()
        if opts.fix_unicode:
            self.fix_unicode()
        if isinstance(opts.extra_transforms, list):
            self.transforms.extend(opts.extra_transforms)
        return self

    def add(self, transform, condition=True) -> TextTransformer:
        if condition:
            self.transforms.append(transform)
        return self

    def transform(self, text: str) -> str:

        for ft in self.transforms:
            text = ft(text)

        return text.strip()

    def fix_hyphenation(self) -> TextTransformer:
        return self.add(TEXT_TRANSFORMS.fix_hyphenation)

    def fix_unicode(self) -> TextTransformer:
        return self.add(TEXT_TRANSFORMS.fix_unicode)

    def fix_whitespaces(self) -> TextTransformer:
        return self.add(TEXT_TRANSFORMS.fix_whitespaces)

    def fix_ftfy(self) -> TextTransformer:
        return self.add(TEXT_TRANSFORMS.fix_ftfy_text)

    def fix_accents(self) -> TextTransformer:
        return self.add(TEXT_TRANSFORMS.fix_accents)
