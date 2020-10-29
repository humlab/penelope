import typing

import nltk

from .extra_stopwords import EXTRA_STOPWORDS, EXTRA_SWEDISH_STOPWORDS

downloader = nltk.downloader.Downloader()

for package in ['punkt', 'stopwords']:
    if not downloader.is_installed(package):
        nltk.download(package)

stopwords = nltk.corpus.stopwords


def extended_stopwords(language: str = 'swedish', extra_stopwords: typing.Set[str] = None) -> typing.Set[str]:
    extra_stopwords = extra_stopwords or EXTRA_SWEDISH_STOPWORDS
    return (set(stopwords.words(language)).union(set(extra_stopwords or [])))

from nltk.tokenize import word_tokenize, sent_tokenize, casual_tokenize
