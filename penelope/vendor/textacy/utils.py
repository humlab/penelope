import collections
import os
from typing import Callable, Dict, Sequence, Union

import spacy.tokens
import textacy
from loguru import logger
from penelope.vendor.spacy import prepend_spacy_path
from spacy import attrs
from spacy.language import Language as SpacyLanguage
from textacy import Corpus as TextacyCorpus
from textacy.representations.vectorizers import Vectorizer


def generate_word_count_score(corpus: textacy.Corpus, normalize: str, count: int, weighting: str = 'count'):
    wc = corpus.word_counts(normalize=normalize, weighting=weighting, as_strings=True)
    d = {i: set([]) for i in range(1, count + 1)}
    for k, v in wc.items():
        if v <= count:
            d[v].add(k)
    return d


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
    corpus: TextacyCorpus,
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
        bow = doc_to_bow(doc, target=normalize, weighting=weighting, as_strings=True, include=include)
        token_counter.update(bow)
        if normalize == 'lemma':
            lower_cased_word_counts = collections.Counter()
            for k, v in token_counter.items():
                lower_cased_word_counts.update({k.lower(): v})
            token_counter = lower_cased_word_counts
    return token_counter.most_common(n_top)


def doc_to_bow(
    doc: spacy.tokens.Doc,
    target: str = 'lemma',
    weighting: str = 'count',
    as_strings: bool = False,
    include: Callable = None,
    n_min_count: int = 2,
):

    weighing_keys = {'count', 'freq'}
    target_keys = {'lemma': attrs.LEMMA, 'lower': attrs.LOWER, 'orth': attrs.ORTH}

    default_exclude = lambda x: x.is_stop or x.is_punct or x.is_space
    exclude = default_exclude if include is None else lambda x: x.is_stop or x.is_punct or x.is_space or not include(x)

    assert weighting in weighing_keys
    assert target in target_keys

    target_weights = doc.count_by(target_keys[target], exclude=exclude)
    n_tokens = doc._.n_tokens

    if weighting == 'freq':
        target_weights = {id_: weight / n_tokens for id_, weight in target_weights.items()}

    store = doc.vocab.strings
    if as_strings:
        bow = {store[word_id]: count for word_id, count in target_weights.items() if count >= n_min_count}
    else:
        bow = target_weights  # { word_id: count for word_id, count in target_weights.items() }

    return bow


def load_term_substitutions(filepath: str, default_term: str = '_gpe_', delim: str = ';', vocab=None) -> dict:

    substitutions = {}

    if not os.path.isfile(filepath):
        return {}

    with open(filepath) as f:
        substitutions = {
            x[0].strip(): x[1].strip()
            for x in (tuple(line.lower().split(delim)) + (default_term,) for line in f.readlines())
            if x[0].strip() != ''
        }

    if vocab is not None:

        extras = {x.norm_: substitutions[x.lower_] for x in vocab if x.lower_ in substitutions}
        substitutions.update(extras)

        extras = {x.lower_: substitutions[x.norm_] for x in vocab if x.norm_ in substitutions}
        substitutions.update(extras)

    substitutions = {k: v for k, v in substitutions.items() if k != v}

    return substitutions


def term_substitutions(data_folder: str, filename: str = 'term_substitutions.txt', vocab=None) -> dict:
    path = os.path.join(data_folder, filename)
    logger.info('Loading term substitution mappings...')
    data = load_term_substitutions(path, default_term='_masked_', delim=';', vocab=vocab)
    return data


def vectorize_terms(terms, vectorizer_args: Dict):
    vectorizer = Vectorizer(**vectorizer_args)
    doc_term_matrix = vectorizer.fit_transform(terms)
    id2word = vectorizer.id_to_term
    return doc_term_matrix, id2word


def save_corpus(
    corpus: textacy.Corpus, filename: str, lang=None, include_tensor: bool = False
):  # pylint: disable=unused-argument
    if not include_tensor:
        for doc in corpus:
            doc.tensor = None
    corpus.save(filename)


def load_corpus(filename: str, lang: Union[str, SpacyLanguage]) -> textacy.Corpus:
    lang: Union[str, SpacyLanguage] = prepend_spacy_path(lang)
    corpus = textacy.Corpus.load(lang, filename)
    return corpus


def merge_named_entities(corpus: textacy.Corpus):
    logger.info('Working: Merging named entities...')
    try:
        for doc in corpus:
            named_entities = textacy.extract.entities(doc)
            textacy.spacier.utils.merge_spans(named_entities, doc)
    except TypeError as ex:
        logger.error(ex)
        logger.info('NER merge failed')
