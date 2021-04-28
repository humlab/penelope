# type: ignore

from typing import Sequence, Set

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
