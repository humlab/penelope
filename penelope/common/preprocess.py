from typing import Callable

from .dehyphenation import SwedishDehyphenator
from .tokenize.sparv_tokenize import SegmenterRepository


class Registry:
    items: dict = {}

    @classmethod
    def get(cls, key: str):
        if key not in cls.items:
            raise ValueError(f"preprocessor {key} is not registered")
        return cls.items.get(key)

    @classmethod
    def register(cls, **args):
        def decorator(fn):
            if args.get("type") == "function":
                fn = fn()
            cls.items[args.get("key") or fn.__name__] = fn
            return fn

        return decorator

    @classmethod
    def is_registered(cls, key: str):
        return key in cls.items


class DehypenatorProvider:
    dehyphenator: SwedishDehyphenator = None

    @classmethod
    def locate(cls) -> SwedishDehyphenator:
        if cls.dehyphenator is None:
            cls.dehyphenator = SwedishDehyphenator()
        return cls.dehyphenator


@Registry.register(key="dehyphen")
def dehyphen(text: str) -> str:
    """Remove hyphens from `text`."""
    dehyphenated_text = DehypenatorProvider.locate().dehyphen_text(text)
    return dehyphenated_text


@Registry.register(key="pretokenize")
def pretokenize(text: str) -> str:
    """Tokenize `text`, then join resulting tokens."""
    return ' '.join(SegmenterRepository.default_tokenize(text))


@Registry.register(key="strip", type="function")
def fx_strip() -> Callable[[str], str]:
    return str.strip


@Registry.register(key="dedent")
def dedent(text: str) -> str:
    """Remove whitespaces before and after newlines"""
    return '\n'.join(x.strip() for x in text.split('\n'))
