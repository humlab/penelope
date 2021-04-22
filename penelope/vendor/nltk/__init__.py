# type: ignore

from typing import Set

import nltk
from nltk.tokenize import casual_tokenize, sent_tokenize, word_tokenize

from .extra_stopwords import EXTRA_STOPWORDS, EXTRA_SWEDISH_STOPWORDS

downloader = nltk.downloader.Downloader()

for package in ['punkt', 'stopwords']:
    if not downloader.is_installed(package):
        nltk.download(package)

stopwords = nltk.corpus.stopwords

STOPWORDS_CACHE = {}


def extended_stopwords(language: str = 'swedish', extra_stopwords: Set[str] = None) -> Set[str]:
    """Returns NLTK stopwords for given lanuage extended with specified extra stopwords"""

    if language not in STOPWORDS_CACHE:
        _stopwords = (
            set(stopwords.words(language)).union(EXTRA_STOPWORDS.get(language, {})).union(set(extra_stopwords or []))
        )
        STOPWORDS_CACHE[language] = _stopwords

    return STOPWORDS_CACHE.get(language, {})
