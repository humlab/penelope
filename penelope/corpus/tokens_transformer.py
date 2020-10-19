from __future__ import annotations

import inspect
from typing import Any, Dict, List

import textacy.preprocessing.remove as textacy_remove

from . import transforms

# pylint: disable=too-many-arguments

DEFAULT_TOKENS_TRANSFORM_OPTIONS = dict(
    only_alphabetic=False,
    only_any_alphanumeric=False,
    to_lower=False,
    to_upper=False,
    min_len=1,
    max_len=100,
    remove_accents=False,
    remove_stopwords=False,
    stopwords=None,
    extra_stopwords=None,
    language="swedish",
    keep_numerals=True,
    keep_symbols=True,
)


def transformer_defaults():
    sig = inspect.signature(TokensTransformer.__init__)
    return {name: param.default for name, param in sig.parameters.items() if param.name != 'self'}


def transformer_defaults_filter(opts: Dict[str, Any]):
    if opts is None:
        return {}
    return {k: v for k, v in opts.items() if k in transformer_defaults()}


class TokensTransformer:
    """Transforms applied on tokenized text"""

    def __init__(
        self,
        only_alphabetic: bool = False,
        only_any_alphanumeric: bool = False,
        to_lower: bool = False,
        to_upper: bool = False,
        min_len: int = None,
        max_len: int = None,
        remove_accents: bool = False,
        remove_stopwords: bool = False,
        stopwords: Any = None,
        extra_stopwords: List[str] = None,
        language: str = "swedish",
        keep_numerals: bool = True,
        keep_symbols: bool = True,
    ):
        self.transforms = []

        self.min_chars_filter(1)

        if to_lower:
            self.to_lower()

        if to_upper:
            self.to_upper()

        if max_len is not None:
            self.max_chars_filter(max_len)

        if keep_symbols is False:
            self.remove_symbols()

        if remove_accents:
            self.remove_accents()

        if min_len is not None and min_len > 1:
            self.min_chars_filter(min_len)

        if only_alphabetic:
            self.only_alphabetic()

        if only_any_alphanumeric:
            self.only_any_alphanumeric()

        if keep_numerals is False:
            self.remove_numerals()

        if remove_stopwords or (stopwords is not None):
            self.remove_stopwords(language_or_stopwords=(stopwords or language), extra_stopwords=extra_stopwords)

    def add(self, transform) -> TokensTransformer:
        self.transforms.append(transform)
        return self

    def transform(self, tokens) -> TokensTransformer:

        for ft in self.transforms:
            tokens = [x for x in ft(tokens)]

        return tokens

    # Shortcuts

    def min_chars_filter(self, n_chars) -> TokensTransformer:
        if (n_chars or 0) < 1:
            return self
        return self.add(transforms.min_chars_filter(n_chars))

    def max_chars_filter(self, n_chars) -> TokensTransformer:
        if (n_chars or 0) < 1:
            return self
        return self.add(transforms.max_chars_filter(n_chars))

    def to_lower(self) -> TokensTransformer:
        return self.add(transforms.lower_transform())

    def to_upper(self) -> TokensTransformer:
        return self.add(transforms.upper_transform())

    def remove_symbols(self) -> TokensTransformer:
        return self.add(transforms.remove_symbols()).add(transforms.min_chars_filter(1))

    def only_alphabetic(self) -> TokensTransformer:
        return self.add(transforms.only_alphabetic_filter())

    def remove_numerals(self) -> TokensTransformer:
        return self.add(transforms.remove_numerals())

    def remove_stopwords(self, language_or_stopwords=None, extra_stopwords=None) -> TokensTransformer:
        if language_or_stopwords is None:
            return self
        return self.add(transforms.remove_stopwords(language_or_stopwords, extra_stopwords))

    def remove_accents(self) -> TokensTransformer:
        return self.add(textacy_remove.remove_accents)

    def only_any_alphanumeric(self) -> TokensTransformer:
        return self.add(transforms.only_any_alphanumeric())
