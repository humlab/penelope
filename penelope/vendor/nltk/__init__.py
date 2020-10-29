import typing

import nltk
from nltk.tokenize import casual_tokenize, sent_tokenize, word_tokenize

from .extra_stopwords import EXTRA_STOPWORDS, EXTRA_SWEDISH_STOPWORDS

downloader = nltk.downloader.Downloader()

for package in ['punkt', 'stopwords']:
    if not downloader.is_installed(package):
        nltk.download(package)

stopwords = nltk.corpus.stopwords


def extended_stopwords(language: str = 'swedish', extra_stopwords: typing.Set[str] = None) -> typing.Set[str]:
    """Returns NLTK stopwords for given lanuage extended with specified extra stopwords"""
    extra_stopwords = extra_stopwords or EXTRA_SWEDISH_STOPWORDS
    return set(stopwords.words(language)).union(set(extra_stopwords or []))
