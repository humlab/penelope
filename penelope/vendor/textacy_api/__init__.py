# type: ignore
# pylint: disable=unused-import

from __future__ import annotations

import ftfy

try:

    from textacy.corpus import Corpus
    from textacy.preprocessing.normalize import unicode as normalize_unicode
    from textacy.preprocessing.normalize import whitespace as normalize_whitespace
    from textacy.preprocessing.replace import currency_symbols as replace_currency_symbols
    from textacy.representations.matrix_utils import filter_terms_by_df
    from textacy.representations.vectorizers import Vectorizer
    from textacy.tm import TopicModel

    __textacy_installed: bool = True

except (ImportError, NameError):

    __textacy_installed: bool = False

    class Vectorizer:
        def __init__(self, **_):
            ...

        @property
        def id_to_term(self) -> dict:
            raise ModuleNotFoundError()

        @property
        def terms_list(self) -> list:
            return []

        def fit(self, *_, **__) -> "Vectorizer":
            raise ModuleNotFoundError()

        def fit_transform(self, *_, **__):
            raise ModuleNotFoundError()

        def transform(self, *_, **__):
            raise ModuleNotFoundError()

    normalize_unicode = ftfy.fix_encoding
    normalize_whitespace = lambda s: " ".join(s.split())
    replace_currency_symbols = lambda s: s

    def filter_terms_by_df(*_, **__):
        return tuple()


try:
    from ._textacy import mdw_modified as mdw
    from ._textacy.mdw_modified import compute_most_discriminating_terms
except (ImportError, NameError):
    ...


try:
    from ._textacy.extract import (
        ExtractPipeline,
        FrequentWordsFilter,
        InfrequentWordsFilter,
        MinCharactersFilter,
        PredicateFilter,
        StopwordFilter,
    )
    from ._textacy.utils import get_most_frequent_words, load_corpus  # frequent_document_words,  infrequent_words
except (ImportError, NameError):
    ...

# try:
#     from textacy.extract.basics import words
# except (ImportError, NameError):

#     def words(*_, **__):
#         return []
