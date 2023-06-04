# type: ignore

from typing import Callable, Iterable, Sequence, Set

import nltk
from nltk.tokenize import casual_tokenize, sent_tokenize, word_tokenize

from .extra_stopwords import EXTRA_STOPWORDS, EXTRA_SWEDISH_STOPWORDS


def download_stopwords():
    downloader = nltk.downloader.Downloader()
    for package in ['punkt', 'stopwords']:
        if not downloader.is_installed(package):
            nltk.download(package)


# def nltk_stopwords():
#     download_stopwords()
#     _stopwords = nltk.corpus.stopwords
#     return _stopwords

STOPWORDS = None


def get_stopwords(language: str) -> Sequence[str]:
    global STOPWORDS
    if STOPWORDS is None:
        download_stopwords()
        STOPWORDS = nltk.corpus.stopwords.words
    return STOPWORDS(language)


STOPWORDS_CACHE = {}


def extended_stopwords(language: str = 'swedish', extra_stopwords: Set[str] = None) -> Set[str]:
    """Returns NLTK stopwords for given lanuage extended with specified extra stopwords"""

    if language not in STOPWORDS_CACHE:
        _stopwords = (
            set(get_stopwords(language)).union(EXTRA_STOPWORDS.get(language, {})).union(set(extra_stopwords or []))
        )
        STOPWORDS_CACHE[language] = _stopwords

    return STOPWORDS_CACHE.get(language, {})


def load_stopwords(language_or_stopwords: str | Iterable[str] = 'swedish', extra_stopwords=None) -> set[str]:
    stopwords = (
        extended_stopwords(language_or_stopwords, extra_stopwords)
        if isinstance(language_or_stopwords, str)
        else set(language_or_stopwords or {}).union(set(extra_stopwords or {}))
    )
    return stopwords


def remove_stopwords_factory(
    language_or_stopwords: str | Iterable[str] = 'swedish', extra_stopwords: Iterable[str] = None
) -> Callable[[Iterable[str]], Iterable[str]]:
    stopwords = load_stopwords(language_or_stopwords, extra_stopwords)
    return lambda tokens: (x for x in tokens if x not in stopwords)
