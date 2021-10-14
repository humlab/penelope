from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from textacy.preprocessing import remove as textacy_remove

from . import transforms
from .token2id import Token2Id

# pylint: disable=too-many-arguments


@dataclass
class TokensTransformOpts:

    to_lower: bool = False
    to_upper: bool = False

    only_alphabetic: bool = False
    only_any_alphanumeric: bool = False
    min_len: int = 1
    max_len: Optional[int] = None
    remove_accents: bool = False
    remove_stopwords: bool = False
    stopwords: Optional[Iterable[str]] = None
    extra_stopwords: Optional[List[str]] = None
    language: Optional[str] = "swedish"
    keep_numerals: bool = True
    keep_symbols: bool = True

    def is_valid(self, attribute: str) -> bool:
        return hasattr(self, attribute)

    @property
    def props(self):
        return {k: v for k, v in self.__dict__.items() if k != 'props' and not k.startswith('_') and not callable(v)}

    def mask(self, tokens: pd.Series, token2id: Token2Id = None) -> np.ndarray:

        mask = np.repeat(True, len(tokens))

        if len(tokens) == 0:
            return mask

        if np.issubdtype(tokens.dtype, np.integer):
            if token2id is None:
                raise ValueError("mask(id): vocabulary is missing")

            tokens = tokens.map(token2id.id2token)

        if self.min_len > 1:
            mask &= tokens.str.len() >= self.min_len

        if self.max_len:
            mask &= tokens.str.len() <= self.max_len

        if self.only_alphabetic:
            mask &= tokens.apply(lambda t: all(c in transforms.ALPHABETIC_CHARS for c in t))

        if self.only_any_alphanumeric:
            mask &= tokens.apply(lambda t: any(c.isalnum() for c in t))

        if self.remove_accents:
            # FIXME Not implemented
            pass

        if self.remove_stopwords:
            mask &= ~tokens.isin(transforms.load_stopwords(self.stopwords, self.extra_stopwords))

        if not self.keep_numerals:
            mask &= ~tokens.str.isnumeric()

        if not self.keep_symbols:
            mask &= ~tokens.apply(lambda t: all(c in transforms.SYMBOLS_CHARS for c in t))

        mask &= tokens != ''

        return mask


DEFAULT_TOKENS_TRANSFORM_OPTIONS = TokensTransformOpts().props


def transformer_defaults():
    sig = inspect.signature(TokensTransformer.__init__)
    return {name: param.default for name, param in sig.parameters.items() if param.name != 'self'}


def transformer_defaults_filter(opts: Dict[str, Any]):
    if opts is None:
        return {}
    return {k: v for k, v in opts.items() if k in transformer_defaults()}


class TokensTransformerBase:
    """Transforms applied on tokenized text"""

    def __init__(self):
        self.transforms = []

    def add(self, transform: Callable[[List[str]], List[str]]) -> TokensTransformer:
        self.transforms.append(transform)
        return self

    def transform(self, tokens: List[str]) -> List[str]:

        for ft in self.transforms:
            tokens = [x for x in ft(tokens)]

        return tokens

    # Shortcuts


class TokensTransformerMixin:
    """Convinient functions that enable chaining of transforms"""

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

        return self.add(textacy_remove.accents)

    def only_any_alphanumeric(self) -> TokensTransformer:
        return self.add(transforms.only_any_alphanumeric())

    def ingest(self, opts: TokensTransformOpts):

        assert isinstance(opts, TokensTransformOpts)
        self.min_chars_filter(1)

        if opts.to_lower:
            self.to_lower()

        if opts.to_upper:
            self.to_upper()

        if opts.max_len is not None:
            self.max_chars_filter(opts.max_len)

        if opts.keep_symbols is False:
            self.remove_symbols()

        if opts.remove_accents:
            self.remove_accents()

        if opts.min_len is not None and opts.min_len > 1:
            self.min_chars_filter(opts.min_len)

        if opts.only_alphabetic:
            self.only_alphabetic()

        if opts.only_any_alphanumeric:
            self.only_any_alphanumeric()

        if opts.keep_numerals is False:
            self.remove_numerals()

        if opts.remove_stopwords or (opts.stopwords is not None):
            self.remove_stopwords(
                language_or_stopwords=(opts.stopwords or opts.language), extra_stopwords=opts.extra_stopwords
            )


# TODO: Refactor to make it more extendable
class TokensTransformer(TokensTransformerMixin, TokensTransformerBase):
    """Transforms applied on tokenized text"""

    def __init__(self, transform_opts: TokensTransformOpts):
        TokensTransformerBase.__init__(self)
        self.ingest(transform_opts)
