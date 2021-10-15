import collections
from typing import Mapping, Sequence, Union

import textacy
from penelope.vendor.spacy import prepend_spacy_path
from spacy.language import Language

from ..spacy import token_count_by


def infrequent_words(
    corpus: textacy.Corpus,
    normalize: str = 'lemma',
    weighting: str = 'count',
    threshold: int = 0,
    as_strings: bool = False,
):
    '''Returns set of infrequent words i.e. words having total count less than given threshold'''

    if weighting == 'count' and threshold <= 1:
        return set([])

    _word_counts = corpus.word_counts(normalize=normalize, weighting=weighting, as_strings=as_strings)
    _words = {w for w in _word_counts if _word_counts[w] < threshold}

    return _words


def frequent_document_words(
    corpus: textacy.Corpus,
    normalize="lemma",
    weighting="freq",
    dfs_threshold=80,
    as_strings=True,
):
    """Returns set of words that occurrs freuently in many documents, candidate stopwords"""
    document_freqs = corpus.word_doc_counts(
        normalize=normalize, weighting=weighting, smooth_idf=True, as_strings=as_strings
    )
    frequent_words = {w for w, f in document_freqs.items() if int(round(f, 2) * 100) >= dfs_threshold}
    return frequent_words


def get_most_frequent_words(
    corpus: textacy.Corpus,
    n_top: int,
    normalize: str = 'lemma',
    include_pos: Sequence[str] = None,
    weighting: str = 'count',
):
    include_pos = include_pos or ['VERB', 'NOUN', 'PROPN']
    include = lambda x: x.pos_ in include_pos
    token_counter = collections.Counter()
    for doc in corpus:
        frequencies: Mapping[str, int] = token_count_by(
            doc=doc, target=normalize, weighting=weighting, as_strings=True, include=include
        )
        # FIXME: if normalize is lemma then make count case-insensitive
        # if normalize == 'lemma':
        #     frequencies = to_lowercased_key_counts(frequencies)
        token_counter.update(frequencies)
    return token_counter.most_common(n_top)


def load_corpus(filename: str, lang: Union[str, Language]) -> textacy.Corpus:
    lang: Union[str, Language] = prepend_spacy_path(lang)
    corpus = textacy.Corpus.load(lang, filename)
    return corpus
