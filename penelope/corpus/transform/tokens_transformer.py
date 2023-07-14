from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd

from penelope.utility.utils import CommaStr
from penelope.vendor.nltk import load_stopwords

from . import transforms as tr
from .transformer import TransformOpts, TransformProperty

if TYPE_CHECKING:
    from ..token2id import Token2Id

# pylint: disable=too-many-arguments


class TokensTransformOpts(TransformOpts):
    registry: tr.TransformRegistry = tr.TokensTransformRegistry
    DEFAULT_TRANSFORMS = CommaStr('')

    to_lower: bool = TransformProperty()
    to_upper: bool = TransformProperty()
    only_alphabetic: bool = TransformProperty()
    only_any_alphanumeric: bool = TransformProperty()

    min_len: int = TransformProperty(alias="min-chars")
    max_len: Optional[int] = TransformProperty(alias="max-chars")

    remove_accents: bool = TransformProperty()

    remove_numerals: bool = TransformProperty()  # alias="remove-numerals", negate=True)
    remove_symbols: bool = TransformProperty()  # alias="remove-symbols", negate=True)

    remove_stopwords: str = TransformProperty()

    def __init__(
        self,
        transforms: str | CommaStr | dict[str, bool] = None,
        extras: list[Callable] = None,
        extra_stopwords: list[str] = None,
    ):
        self.extra_stopwords: Optional[list[str]] = extra_stopwords
        if extra_stopwords:
            extras = extras or []
            extras.append(lambda t: (x for x in t if x not in extra_stopwords))
        super().__init__(transforms=transforms, extras=extras)

    @property
    def has_effect(self) -> bool:
        return self.transforms != "" or self.extras

    @property
    def of_no_effect(self):
        return not self.has_effect

    @property
    def props(self):
        return {k: v for k, v in self.__dict__.items() if k != 'props' and not k.startswith('_') and not callable(v)}

    def mask(self, tokens: pd.Series, token2id: Token2Id = None) -> np.ndarray:
        return create_tagged_frame_mask(self, tokens, token2id)


def create_tagged_frame_mask(opts: TokensTransformOpts, tokens: pd.Series, token2id: Token2Id = None) -> np.ndarray:
    mask = np.repeat(True, len(tokens))

    if len(tokens) == 0:
        return mask

    if np.issubdtype(tokens.dtype, np.integer):
        if token2id is None:
            raise ValueError("mask(id): vocabulary is missing")

        tokens = tokens.apply(token2id.id2token.get)

    if opts.min_len > 1:
        mask &= tokens.str.len() >= opts.min_len

    if opts.max_len:
        mask &= tokens.str.len() <= opts.max_len

    if opts.only_alphabetic:
        mask &= tokens.apply(lambda t: all(c in tr.ALPHABETIC_CHARS for c in t))

    if opts.only_any_alphanumeric:
        mask &= tokens.apply(lambda t: any(c.isalnum() for c in t))

    if opts.remove_accents:
        # FIXME Not implemented
        pass

    if opts.remove_stopwords or opts.extra_stopwords:
        mask &= ~tokens.isin(load_stopwords(opts.remove_stopwords, opts.extra_stopwords))

    if opts.remove_numerals:
        mask &= ~tokens.str.isnumeric()

    if opts.remove_symbols:
        mask &= ~tokens.apply(lambda t: all(c in tr.SYMBOLS_CHARS for c in t))

    mask &= tokens != ''

    return mask


# class TokensTransformer:
#     """Transforms applied on tokenized text"""

#     def __init__(self, transform_opts: TokensTransformOpts):
#         self.registry: tr.TokensTransformRegistry = tr.TokensTransformRegistry
#         if not isinstance(transform_opts, TokensTransformOpts):
#             raise TypeError(f"Expected {TokensTransformOpts}, got {type(transform_opts)}")
#         self.transform_opts: TokensTransformOpts = transform_opts
#         self.gfx = None

#     def transform(self, tokens: list[str]) -> list[str]:
#         if self.gfx is None:
#             self.gfx = self.transform_opts.getfx()
#         return self.gfx(tokens)
