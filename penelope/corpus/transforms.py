import re
import string
import unicodedata
from enum import IntEnum, unique
from typing import Callable, Iterable, Set, Union

import ftfy
import nltk

import penelope.vendor.nltk as nltk_utility
from penelope.vendor.textacy_api import normalize_whitespace

# pylint: disable=W0601,E0602

ALPHABETIC_LOWER_CHARS = string.ascii_lowercase + "åäöéàáâãäåæèéêëîïñôöùûÿ"
ALPHABETIC_CHARS = set(ALPHABETIC_LOWER_CHARS + ALPHABETIC_LOWER_CHARS.upper())
SYMBOLS_CHARS = set("'\\¢£¥§©®°±øæç•›€™").union(set('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
ACCENT_CHARS = set('\'`')
SYMBOLS_TRANSLATION = dict.fromkeys(map(ord, SYMBOLS_CHARS), None)
default_tokenizer = nltk.word_tokenize

DEFAULT_HYPHEN_REGEXP = r'\b(\w+)[-¬]\s*\r?\n\s*(\w+)\s*\b'
RE_HYPHEN_REGEXP: re.Pattern = re.compile(DEFAULT_HYPHEN_REGEXP, re.UNICODE)

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

"""Cretae translations that maps characters to first character in each string"""
SPECIAL_CHARS_GROUP_TRANSLATIONS = {k: str.maketrans(v[1:], v[0] * (len(v) - 1)) for k, v in SPECIAL_CHARS.items()}

ALL_IN_ONE_TRANSLATION = str.maketrans(
    *[
        '‐‑⁃‒–—―−－⁻＋⁺⁄∕˜⁓∼∽∿〜～’՚Ꞌꞌ＇‘’‚‛“”„‟´″‴‵‶‷⁗',
        '----------++//~~~~~~~\'\'\'\'\'\'\'\'\'""""`′′′′′′',
    ]
)


def normalize_characters(text: str, groups: str = None) -> str:

    if groups is None:
        return text.translate(ALL_IN_ONE_TRANSLATION)

    for group in groups.split(","):
        text = text.translate(SPECIAL_CHARS_GROUP_TRANSLATIONS[group])

    return text


TokensTransformerFunction = Callable[[Iterable[str]], Iterable[str]]


def load_stopwords(language_or_stopwords: Union[str, Iterable[str]] = 'swedish', extra_stopwords=None) -> Set[str]:
    stopwords = (
        nltk_utility.extended_stopwords(language_or_stopwords, extra_stopwords)
        if isinstance(language_or_stopwords, str)
        else set(language_or_stopwords or {}).union(set(extra_stopwords or {}))
    )
    return stopwords


def remove_empty_filter():
    return lambda t: (x for x in t if x != '')


def remove_hyphens(text: str) -> str:
    result = RE_HYPHEN_REGEXP.sub(r"\1\2\n", text)
    return result


def remove_hyphens_fx(text: str, hyphen_regexp: str = DEFAULT_HYPHEN_REGEXP) -> Callable[[str], str]:
    expr = re.compile(hyphen_regexp, re.UNICODE)
    return lambda t: re.sub(expr, r"\1\2\n", text)


def has_alpha_filter() -> TokensTransformerFunction:
    return lambda tokens: (x for x in tokens if any(map(lambda x: x.isalpha(), x)))


def only_any_alphanumeric() -> TokensTransformerFunction:
    return lambda tokens: (t for t in tokens if any(c.isalnum() for c in t))


def only_alphabetic_filter() -> TokensTransformerFunction:
    return lambda tokens: (x for x in tokens if any(c in x for c in ALPHABETIC_CHARS))


def remove_stopwords(
    language_or_stopwords: Union[str, Iterable[str]] = 'swedish', extra_stopwords: Iterable[str] = None
) -> TokensTransformerFunction:
    stopwords = load_stopwords(language_or_stopwords, extra_stopwords)
    return lambda tokens: (x for x in tokens if x not in stopwords)  # pylint: disable=W0601,E0602


def min_chars_filter(n_chars: int = 3) -> TokensTransformerFunction:
    return lambda tokens: (x for x in tokens if len(x) >= n_chars)


def max_chars_filter(n_chars: int = 3):
    return lambda tokens: (x for x in tokens if len(x) <= n_chars)


def lower_transform() -> TokensTransformerFunction:
    return lambda tokens: map(lambda y: y.lower(), tokens)


def upper_transform() -> TokensTransformerFunction:
    return lambda tokens: map(lambda y: y.upper(), tokens)


def remove_numerals() -> TokensTransformerFunction:
    return lambda tokens: (x for x in tokens if not x.isnumeric())


def remove_symbols() -> TokensTransformerFunction:
    return lambda tokens: (x.translate(SYMBOLS_TRANSLATION) for x in tokens)


def strip_accents(text: str) -> str:
    """https://stackoverflow.com/a/44433664/12383895"""
    text: str = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)


# @deprecated
# def remove_accents() -> TokensTransformerFunction:
#     return lambda tokens: (x.translate(SYMBOLS_TRANSLATION) for x in tokens)


class TEXT_TRANSFORMS:
    fix_hyphenation = remove_hyphens
    fix_unicode = lambda text: unicodedata.normalize("NFC", text)
    fix_whitespaces = normalize_whitespace
    fix_accents = strip_accents
    fix_currency_symbols = lambda text: RE_CURRENCY_SYMBOLS.sub("__cur__", text)
    fix_ftfy_text = ftfy.fix_text
    fix_encoding = ftfy.fix_encoding


@unique
class KnownTransformType(IntEnum):
    fix_hyphenation = 1
    fix_unicode = 2
    fix_whitespaces = 3
    fix_accents = 4
    fix_currency_symbols = 5
    fix_ftfy_text = 6
    fix_ftfy_fix_encoding = 7

    @property
    def transform(self):
        return getattr(TEXT_TRANSFORMS, self.name)
