# type: ignore
# pylint: disable=unused-import
# flake8: noqa

from .extract import (
    ExtractPipeline,
    FrequentWordsFilter,
    InfrequentWordsFilter,
    MinCharactersFilter,
    PredicateFilter,
    StopwordFilter,
)
from .fallbacks import Corpus, TopicModel, Vectorizer, filter_terms_by_df, normalize_whitespace
from .mdw_modified import compute_most_discriminating_terms, most_discriminating_terms
from .utils import get_most_frequent_words, load_corpus

try:
    from textacy.corpus import Corpus
    from textacy.preprocessing.normalize import whitespace as normalize_whitespace
    from textacy.representations.matrix_utils import filter_terms_by_df
    from textacy.representations.vectorizers import Vectorizer
    from textacy.tm import TopicModel

except (ImportError, NameError):
    ...
