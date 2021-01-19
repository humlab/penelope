# -*- coding: utf-8 -*-
import logging
import re
import string

import nltk
import penelope.vendor.nltk as nltk_utility

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

ALPHABETIC_LOWER_CHARS = string.ascii_lowercase + "åäöéàáâãäåæèéêëîïñôöùûÿ"
ALPHABETIC_CHARS = set(ALPHABETIC_LOWER_CHARS + ALPHABETIC_LOWER_CHARS.upper())
SYMBOLS_CHARS = set("'\\¢£¥§©®°±øæç•›€™").union(set('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
SYMBOLS_TRANSLATION = dict.fromkeys(map(ord, SYMBOLS_CHARS), None)
default_tokenizer = nltk.word_tokenize

# pylint: disable=W0601,E0602


def remove_empty_filter():
    return lambda t: (x for x in t if x != '')


hyphen_regexp = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)


def remove_hyphens(text: str) -> str:
    result = re.sub(hyphen_regexp, r"\1\2\n", text)
    return result


def has_alpha_filter():
    return lambda tokens: (x for x in tokens if any(map(lambda x: x.isalpha(), x)))


def only_any_alphanumeric():
    return lambda tokens: (t for t in tokens if any(c.isalnum() for c in t))


def only_alphabetic_filter():
    return lambda tokens: (x for x in tokens if any(c in x for c in ALPHABETIC_CHARS))


def remove_stopwords(language_or_stopwords='swedish', extra_stopwords=None):
    if isinstance(language_or_stopwords, str):
        stopwords = set(nltk_utility.stopwords.words(language_or_stopwords))
    else:
        stopwords = set(language_or_stopwords or {})
    stopwords = stopwords.union(set(extra_stopwords or {}))
    return lambda tokens: (x for x in tokens if x not in stopwords)  # pylint: disable=W0601,E0602


def min_chars_filter(n_chars: int = 3):
    return lambda tokens: (x for x in tokens if len(x) >= n_chars)


def max_chars_filter(n_chars: int = 3):
    return lambda tokens: (x for x in tokens if len(x) <= n_chars)


def lower_transform():
    return lambda tokens: map(lambda y: y.lower(), tokens)


def upper_transform():
    return lambda tokens: map(lambda y: y.upper(), tokens)


def remove_numerals():
    return lambda tokens: (x for x in tokens if not x.isnumeric())


def remove_symbols():
    return lambda tokens: (x.translate(SYMBOLS_TRANSLATION) for x in tokens)


def remove_accents():
    return lambda tokens: (x.translate(SYMBOLS_TRANSLATION) for x in tokens)
