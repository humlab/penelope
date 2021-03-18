import collections
import itertools
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import penelope.utility as utility
import spacy.tokens
import textacy
from penelope.corpus import DocumentIndex
from spacy import attrs
from textacy.vsm import Vectorizer

logger = utility.getLogger()

POS_TO_COUNT = {
    'SYM': 0,
    'PART': 0,
    'ADV': 0,
    'NOUN': 0,
    'CCONJ': 0,
    'ADJ': 0,
    'DET': 0,
    'ADP': 0,
    'INTJ': 0,
    'VERB': 0,
    'NUM': 0,
    'PRON': 0,
    'PROPN': 0,
}

POS_NAMES = list(sorted(POS_TO_COUNT.keys()))


def generate_word_count_score(corpus: textacy.Corpus, normalize: str, count: int, weighting: str = 'count'):
    wc = corpus.word_counts(normalize=normalize, weighting=weighting, as_strings=True)
    d = {i: set([]) for i in range(1, count + 1)}
    for k, v in wc.items():
        if v <= count:
            d[v].add(k)
    return d


def generate_word_document_count_score(corpus: textacy.Corpus, normalize: str, threshold: int = 75):
    wc = corpus.word_doc_counts(normalize=normalize, weighting='freq', smooth_idf=True, as_strings=True)
    d = {i: set([]) for i in range(threshold, 101)}
    for k, v in wc.items():
        slot = int(round(v, 2) * 100)
        if slot >= threshold:
            d[slot].add(k)
    return d


def count_documents_by_pivot(corpus: textacy.Corpus, attribute: str) -> List[int]:
    """Return a list of document counts per group defined by attribute
    Assumes documents are sorted by attribute!
    """

    def fx_key(doc: spacy.tokens.Doc):
        return doc._.meta[attribute]

    return [len(list(g)) for _, g in itertools.groupby(corpus, fx_key)]


def count_documents_in_index_by_pivot(document_index: DocumentIndex, attribute: str) -> List[int]:
    """Return a list of document counts per group defined by attribute
    Assumes documents are sorted by attribute!
    Same as count_documents_by_pivot but uses document index instead of (spaCy) corpus
    """
    assert document_index[attribute].is_monotonic_increasing, 'Documents *MUST* be sorted by TIME-SLICE attribute!'
    # TODO: Either sort documents (and corpus or term stream!) prior to this call - OR force sortorder by filename (i.e add year-prefix)
    return list(document_index.groupby(attribute).size().values)


def get_document_by_id(
    corpus: textacy.Corpus, document_id: Tuple[int, str], field_name: str = 'document_id'
) -> spacy.tokens.Doc:
    for doc in corpus.get(lambda x: x._.meta[field_name] == document_id, limit=1):
        return doc
    assert False, 'Document {} not found in corpus'.format(document_id)
    return None


def get_disabled_pipes_from_filename(filename: str):
    re_pipes = r'^.*disable\((.*)\).*$'
    x = re.match(re_pipes, filename).groups(0)
    if len(x or []) > 0:
        return x[0].split(',')
    return None


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

    token_counter = corpus.word_counts(normalize=normalize, weighting=weighting, as_strings=as_strings)
    words = {w for w in token_counter if token_counter[w] < threshold}

    return words


def frequent_document_words(
    corpus: textacy.Corpus,
    normalize: str = 'lemma',
    weighting: str = 'freq',
    dfs_threshold: int = 80,
    as_strings: bool = True,
):  # pylint: disable=unused-argument
    '''Returns set of words that occurrs freuently in many documents, candidate stopwords'''
    document_freqs = corpus.word_doc_counts(normalize=normalize, weighting=weighting, smooth_idf=True, as_strings=True)
    result = {w for w, f in document_freqs.items() if int(round(f, 2) * 100) >= dfs_threshold}
    return result


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


def get_pos_statistics(doc: spacy.tokens.Doc):
    pos_iter = (x.pos_ for x in doc if x.pos_ not in ['NUM', 'PUNCT', 'SPACE'])
    pos_counts = dict(collections.Counter(pos_iter))
    stats = utility.extend(dict(POS_TO_COUNT), pos_counts)
    return stats


def get_corpus_data(
    corpus: textacy.Corpus, document_index: DocumentIndex, title: str, columns_of_interest: List[str] = None
) -> pd.DataFrame:
    metadata = [
        utility.extend({}, dict(document_id=doc._.meta['document_id']), get_pos_statistics(doc)) for doc in corpus
    ]
    df = pd.DataFrame(metadata)[['document_id'] + POS_NAMES]
    if columns_of_interest is not None:
        document_index = document_index[columns_of_interest]
    df = pd.merge(df, document_index, left_on='document_id', right_on='document_id', how='inner')
    df['title'] = df[title]
    df['words'] = df[POS_NAMES].apply(sum, axis=1)
    return df


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


def _doc_token_stream(doc: spacy.tokens.Doc) -> Iterable[Dict[str, Any]]:
    return (
        dict(
            i=t.i,
            token=t.lower_,
            lemma=t.lemma_,
            pos=t.pos_,
            year=doc._.meta['year'],
            document_id=doc._.meta['document_id'],
        )
        for t in doc
    )


def store_tokens(corpus: textacy.Corpus, filename: str):

    tokens: pd.DataFrame = pd.DataFrame(list(itertools.chain.from_iterable(_doc_token_stream(d) for d in corpus)))

    if filename.endswith('.xlxs'):
        tokens.to_excel(filename)
    else:
        translation_table = str.maketrans({'\t': ' ', '\n': ' ', '"': ' '})
        tokens['token'] = tokens.token.str.translate(translation_table)
        tokens['lemma'] = tokens.lemma.str.translate(translation_table)
        tokens.to_csv(filename, sep='\t')
