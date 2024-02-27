import functools
import re
import string
import unicodedata
from typing import Any, Iterable, Protocol

import ftfy
import nltk
import regex

import penelope.vendor.nltk as nltk_utility
from penelope.common.dehyphenation import SwedishDehyphenator
from penelope.common.tokenize import default_tokenize as _sparv_tokenize
from penelope.vendor.textacy_api import normalize_whitespace

# pylint: disable=W0601,E0602

ALPHABETIC_LOWER_CHARS = string.ascii_lowercase + "åäöéàáâãäåæèéêëîïñôöùûÿ"
ALPHABETIC_UPPER_CHARS = string.ascii_uppercase + "A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝ"
ALPHABETIC_CHARS = set(ALPHABETIC_LOWER_CHARS + ALPHABETIC_LOWER_CHARS.upper())
SYMBOLS_CHARS = set("'\\¢£¥§©®°±øæç•›€™").union(set('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
ACCENT_CHARS = set('\'`')
SYMBOLS_TRANSLATION = dict.fromkeys(map(ord, SYMBOLS_CHARS), None)
default_tokenizer = nltk.word_tokenize

RE_HYPHEN_REGEXP: re.Pattern = re.compile(r'\b(\w+)[-¬]\s*\r?\n\s*(\w+)\s*\b', re.UNICODE)

CURRENCY_SYMBOLS = ''.join(chr(i) for i in range(0xFFFF) if unicodedata.category(chr(i)) == 'Sc')
RE_CURRENCY_SYMBOLS: re.Pattern = re.compile(rf"[{CURRENCY_SYMBOLS}]")

SPECIAL_CHARS = {
    'hyphens': '-‐‑⁃‒–—―',
    'minuses': '-−－⁻',
    'pluses': '+＋⁺',
    'slashes': '/⁄∕',
    'tildes': '~˜⁓∼∽∿〜～',
    'apostrophes': "'’՚Ꞌꞌ＇",
    'single_quotes': "'‘’‚‛",
    'double_quotes': '"“”„‟',
    'accents': '`´',
    'primes': '′″‴‵‶‷⁗',
    'accented_chars': "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ",
}

# SPECIAL_CHARS_ESCAPED = {
#     'hyphens': '-\u2010\u2011\u2043\u2012\u2013\u2014\u2015',
#     'minuses': '-\u2212\uff0d\u207b',
#     'pluses': '+\uff0b\u207a',
#     'slashes': '/\u2044\u2215',
#     'tildes': '~\u02dc\u2053\u223c\u223d\u223f\u301c\uff5e',
#     'apostrophes': "'\u2019\u055a\ua78b\ua78c\uff07",
#     'single_quotes': "'\u2018\u2019\u201a\u201b",
#     'double_quotes': '"\u201c\u201d\u201e\u201f',
#     'accents': '`\xb4',
#     'primes': '\u2032\u2033\u2034\u2035\u2036\u2037\u2057',
# }

# ALL_IN_ONE_TRANSLATION = str.maketrans(
#     *list(map(''.join, zip(*[(v[1:], v[0] * (len(v) - 1)) for _, v in SPECIAL_CHARS.items()])))
# )

"""Create translations that maps characters to first character in each string"""
SPECIAL_CHARS_GROUP_TRANSLATIONS = {k: str.maketrans(v[1:], v[0] * (len(v) - 1)) for k, v in SPECIAL_CHARS.items()}

ALL_IN_ONE_TRANSLATION = str.maketrans(
    *[
        '‐‑⁃‒–—―−－⁻＋⁺⁄∕˜⁓∼∽∿〜～’՚Ꞌꞌ＇‘’‚‛“”„‟´″‴‵‶‷⁗',
        '----------++//~~~~~~~\'\'\'\'\'\'\'\'\'""""`′′′′′′',
    ]
)

pattern = re.compile(r'\.([A-Z])')


# Define a function that uses the compiled pattern to replace occurrences
def insert_space_after_period(text):
    # Use the compiled pattern's sub method for replacement
    return pattern.sub(r'. \1', text)


def normalize_characters(text: str, groups: str = None) -> str:
    if groups is None:
        return text.translate(ALL_IN_ONE_TRANSLATION)

    for group in groups.split(","):
        text = text.translate(SPECIAL_CHARS_GROUP_TRANSLATIONS[group])

    return text


class TokensTransform(Protocol):
    def __call__(self, _: Iterable[str]) -> Iterable[str]: ...


class TextTransform(Protocol):
    def __call__(self, _: str) -> str: ...


Transform = TokensTransform | TextTransform


class DehypenatorProvider:
    dehyphenator: SwedishDehyphenator = None

    @classmethod
    def locate(cls) -> SwedishDehyphenator:
        if cls.dehyphenator is None:
            cls.dehyphenator = SwedishDehyphenator()
        return cls.dehyphenator


class TransformRegistry:
    _items: dict[str, Any] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def add(cls, fn: Any, key: str = None) -> Transform:
        """Add transform function"""
        if key is None:
            key = fn.__name__
        keys = [k.strip() for k in key.replace('_', '-').split(',')]
        cls._items[keys[0]] = fn
        if len(keys) > 1:
            for k in keys[1:]:
                if k in cls._items:
                    continue
                cls._aliases[k] = keys[0]
        return fn

    @classmethod
    def get(cls, key: str) -> Transform:
        """Get transform function by key"""

        if not isinstance(key, str):
            raise ValueError(f"Invalid key type: {type(key).__name__}")

        key = key.replace('_', '-').strip()

        if key in cls._items:
            return cls._items.get(key)

        if key in cls._aliases:
            return cls._items.get(cls._aliases[key])

        if '?' in key:
            """Transform has arguments"""
            return cls.add(fn=cls.resolve_fx(key), key=key)

        raise ValueError(f"preprocessor {key} is not registered")

    @classmethod
    def resolve_fx(cls, key: str | Transform, overrides: dict[str, Transform] = None) -> Transform:
        """Transform functions needs or accepts extra with arguments
        Examples:
            'min-chars?chars=2' => kwargs = {'chars': '2'}
            'min-chars?2'       => args = '2'
            'any-option?[2,3]'  => args = '[2,3]'

        """
        if callable(key):
            return key

        if not isinstance(key, str):
            raise ValueError(f"Invalid key type: {type(key).__name__}")

        overrides = overrides or {}
        key, args = key.split('?', 1) if '?' in key else (key, None)
        key = key.replace('_', '-')

        fx = cls.get(key) if key not in overrides else overrides.get(key)

        if not args:
            return fx

        if '[' in args:
            """args is a list"""
            # FIXME: Cannot use ',' in list since it is used to separate keys
            args = [x.strip() for x in args.lstrip('[').rstrip(']').split(',')]
            return lambda x: fx(x, *args)  # pylint: disable=unnecessary-lambda

        if '=' in args:
            kwargs: dict = {k: v for k, v in [x.split('=') for x in args.split('&')]}
            # kwargs = dict(x.split('=') for x in args.split('&'))
            return lambda x: fx(x, **kwargs)  # pylint: disable=unnecessary-lambda

        return lambda x: fx(x, args)

    @classmethod
    def gets(cls, *keys: tuple[str]) -> list[Transform]:
        return [cls.get(k) for key in keys for k in key.split(',')]

    @classmethod
    def getfx(cls, *keys: tuple[str], extras: list = None, overrides: dict[str, Transform] = None) -> Transform:
        """Get transform function by resolving list of keys into a single function.
        If extras is provided, it will be appended to the list of functions resolved by the keys.
        A key can be a string or a list of strings or a function."""
        keys_or_fxs: list[Transform] = [
            k if isinstance(k, str) else k for key in keys for k in key.split(',') if k if keys
        ]
        fxs: list[Transform] = [cls.resolve_fx(k, overrides=overrides) for k in keys_or_fxs]

        if extras:
            fxs.extend(extras)
        if not fxs:
            return lambda x: x
        return cls.reduce(fxs)

    @classmethod
    def resolve(cls, keys: list[str], extras: list = None, overrides: dict[str, Transform] = None) -> Transform:
        return cls.getfx(*keys, extras=extras, overrides=overrides)

    @classmethod
    def reduce(cls, fxs: list[Transform]) -> Transform:
        return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(fxs))

    @classmethod
    def register(cls, key: str = None, kind: str = None, **args):
        def decorator(fn):
            if kind == "function":
                fn = fn(**args)
            cls.add(fn, key)
            return fn

        return decorator

    @classmethod
    def is_registered(cls, key: str):
        try:
            return cls.get(key) is not None
        except ValueError:
            return False

    @classmethod
    def __getitem__(cls, key: str) -> Transform:
        return cls.get(key)

    @classmethod
    def __contains__(cls, key: str) -> bool:
        return cls.is_registered(key)


class TextTransformRegistry(TransformRegistry):
    _items: dict[str, Any] = {
        'normalize-characters': normalize_characters,
        'normalize-whitespace': normalize_whitespace,
        'ftfy-fix-text': ftfy.fix_text,
        'ftfy-fix-encoding': ftfy.fix_encoding,
        'uppercase': str.upper,
        'to-upper': str.upper,
        'lowercase': str.lower,
        'to-lower': str.lower,
        'strip': str.strip,
    }
    _aliases: dict[str, str] = {
        'fix-whitespaces': 'normalize-whitespace',
        'normalize-whitespaces': 'normalize-whitespace',
        'fix-text': 'ftfy-fix-text',
        'fix-encoding': 'ftfy-fix-encoding',
    }


class TokensTransformRegistry(TransformRegistry):
    _items: dict[str, Any] = {}
    _aliases: dict[str, str] = {}


RE_PERIOD_UPPERCASE_LETTER: re.Pattern = re.compile(r'\.([0-9A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝ\-\–])', re.UNICODE)
RE_SPACE_PERIOD_UPPERCASE_LETTER: re.Pattern = re.compile(
    r'\s\.([0-9A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝ\-\–])', flags=re.UNICODE | re.IGNORECASE
)
RE_SPACE_PERIOD_UPPERCASE_APOSTOPHE: re.Pattern = re.compile(
    r"\s'([0-9A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝ\-\–])", flags=re.UNICODE | re.IGNORECASE
)
# RE_PERIOD_UPPERCASE: re.Pattern = re.compile(r'\.([0-9A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝ\-\–])', re.UNICODE)


@TextTransformRegistry.register(key='space-after-period-uppercase,space-after-sentence,fix-space-after-sentence')
def space_after_period_uppercase(text: str) -> str:
    """Insert space after a period if it is followed by an uppercase letter"""
    text: str = re.sub(RE_PERIOD_UPPERCASE_LETTER, r'. \1', text)
    text: str = re.sub(RE_SPACE_PERIOD_UPPERCASE_LETTER, r' . \1', text)
    text: str = re.sub(RE_SPACE_PERIOD_UPPERCASE_APOSTOPHE, r" ' \1", text)
    return text


@TextTransformRegistry.register(key='dehyphen,fix-hyphenation')
def dehyphen(text: str) -> str:
    result = RE_HYPHEN_REGEXP.sub(r"\1\2\n", text)
    return result


@TextTransformRegistry.register(key='swe-dehyphen,swedish-dehyphen')
def swedish_dehyphen(text: str) -> str:
    """Remove hyphens from `text`."""
    dehyphenated_text = DehypenatorProvider.locate().dehyphen_text(text)
    return dehyphenated_text


@TextTransformRegistry.register(key="dedent")
def dedent(text: str) -> str:
    """Remove whitespaces before and after newlines"""
    return '\n'.join(map(str.strip, text.split('\n')))


@TextTransformRegistry.register(key="strip-accents,fix-accents,remove-accents")
def strip_accents(text: str) -> str:
    """https://stackoverflow.com/a/44433664/12383895"""
    text: str = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)


@TextTransformRegistry.register(key="normalize-unicode")
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


@TextTransformRegistry.register(key='sparv-tokenize')
def sparv_tokenize(text: str) -> list[str]:
    return _sparv_tokenize(text)


@TextTransformRegistry.register(key="replace-currency-symbols")
def replace_currency_symbols(text: str, marker: str = "__cur__") -> str:
    return RE_CURRENCY_SYMBOLS.sub(marker, text)


@TextTransformRegistry.register(key="replace-special-characters")
def replace_special_characters(text: str, replace_character: str = ' ') -> str:
    return regex.sub(r'[^\w\s\p{P}\p{L}]', replace_character, text)


""" Token transforms """


@TokensTransformRegistry.register(key="has-alphabetic")
def has_alphabetic(tokens: Iterable[str]) -> Iterable[str]:
    return (x for x in tokens if any(map(lambda x: x.isalpha(), x)))


@TokensTransformRegistry.register(key="only-any-alphanumeric")
def only_any_alphanumeric(tokens: Iterable[str]) -> Iterable[str]:
    return (t for t in tokens if any(c.isalnum() for c in t))


@TokensTransformRegistry.register(key="only-alphabetic")
def only_alphabetic(tokens: Iterable[str]) -> Iterable[str]:
    return (x for x in tokens if any(c in x for c in ALPHABETIC_CHARS))


# @TokensTransformRegistry.register(key="remove-english-stopwords", kind="function")
# def remove_stopwords_english() -> TokensTransform:
#     return nltk_utility.remove_stopwords_factory(language_or_stopwords='english')


# @TokensTransformRegistry.register(key="remove-swedish-stopwords", kind="function")
# def remove_stopwords_swedish() -> TokensTransform:
#     return nltk_utility.remove_stopwords_factory(language_or_stopwords='swedish')


@TokensTransformRegistry.register(key="remove-stopwords")
def remove_stopwords(tokens: Iterable[str], language: str = 'swedish') -> Iterable[str]:
    return (x for x in tokens if x not in nltk_utility.load_stopwords(language))


def min_chars_factory(n_chars: int = 3) -> TokensTransform:
    return lambda tokens: (x for x in tokens if len(x) >= n_chars)


@TokensTransformRegistry.register(key="min-chars,min-len")
def min_chars(tokens: Iterable[str], chars: int = 3) -> Iterable[str]:
    return (x for x in tokens if len(x) >= int(chars))


@TokensTransformRegistry.register(key="max-chars,max-len")
def max_chars(tokens: Iterable[str], chars: int = 3) -> Iterable[str]:
    return (x for x in tokens if len(x) <= int(chars))


def max_chars_factory(chars: int = 3) -> TokensTransform:
    return lambda tokens: (x for x in tokens if len(x) <= int(chars))


@TokensTransformRegistry.register(key="to-lower,lowercase")
def to_lower(tokens: Iterable[str]) -> Iterable[str]:
    return map(str.lower, tokens)


@TokensTransformRegistry.register(key="to-upper,uppercase")
def to_upper(tokens: Iterable[str]) -> Iterable[str]:
    """Upper case"""
    return map(str.upper, tokens)


@TokensTransformRegistry.register(key="remove-numerals")
def remove_numerals(tokens: Iterable[str]) -> Iterable[str]:
    """Remove numerals"""
    return (x for x in tokens if not x.isnumeric())


@TokensTransformRegistry.register(key="remove-symbols")
def remove_symbols(tokens: Iterable[str]) -> Iterable[str]:
    """Remove symbols"""
    return (t for t in (x.translate(SYMBOLS_TRANSLATION) for x in tokens) if t != '')


@TokensTransformRegistry.register(key="remove-empty")
def remove_empty(tokens: Iterable[str]) -> Iterable[str]:
    """Remove empty tokens"""
    return (x for x in tokens if x != '')
