from __future__ import annotations

import types
from typing import Callable, List

import ftfy
import textacy.preprocessing as preprocessing

TRANSFORMS = types.SimpleNamespace(
    fix_hyphenation=preprocessing.normalize_hyphenated_words,
    fix_unicode=preprocessing.normalize_unicode,
    fix_whitespaces=preprocessing.normalize_whitespace,
    fix_accents=preprocessing.remove_accents,
    fix_currency_symbols=preprocessing.replace_currency_symbols,
    fix_ftfy_text=ftfy.fix_text,
)


class TextTransformer:
    """Transforms applied on non-tokenized text"""

    def __init__(self, transforms: List[Callable] = None):
        self.transforms: List[Callable] = transforms or []

    def add(self, transform, condition=True) -> TextTransformer:
        if condition:
            self.transforms.append(transform)
        return self

    def transform(self, text: str) -> str:

        for ft in self.transforms:
            text = ft(text)

        return text.strip()

    def fix_hyphenation(self) -> TextTransformer:
        return self.add(TRANSFORMS.fix_hyphenation)

    def fix_unicode(self) -> TextTransformer:
        return self.add(TRANSFORMS.fix_unicode)

    def fix_whitespaces(self) -> TextTransformer:
        return self.add(TRANSFORMS.fix_whitespace)
