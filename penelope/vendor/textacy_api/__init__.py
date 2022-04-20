# type: ignore
# pylint: disable=unused-import

from __future__ import annotations

from ._textacy import (  # frequent_document_words,  infrequent_words
    Corpus,
    ExtractPipeline,
    FrequentWordsFilter,
    InfrequentWordsFilter,
    MinCharactersFilter,
    PredicateFilter,
    StopwordFilter,
    TopicModel,
    Vectorizer,
    compute_most_discriminating_terms,
    filter_terms_by_df,
    get_most_frequent_words,
    load_corpus,
    most_discriminating_terms,
    normalize_whitespace,
)

try:
    import textacy

    TEXTACY_INSTALLED: bool = True
except (ImportError, NameError):
    TEXTACY_INSTALLED: bool = False
