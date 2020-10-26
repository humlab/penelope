import os
import random
from collections import defaultdict

import numpy as np
import pytest

from penelope.co_occurrence import (
    WindowsCorpus,
    partitioned_corpus_concept_co_occurrence,
    to_co_ocurrence_matrix,
    to_dataframe,
)
from penelope.co_occurrence.windows_co_occurrence import corpus_concept_co_occurrence
from penelope.corpus import CorpusVectorizer, TokenizedCorpus
from penelope.corpus.readers import InMemoryReader
from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus
from penelope.scripts.concept_co_occurrence import cli_concept_co_occurrence
from penelope.utility.utils import dataframe_to_tuples

jj = os.path.join

TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/tranströmer_corpus_export.csv.zip'

# http://www.nltk.org/howto/collocations.html
# PMI

SIMPLE_CORPUS_ABCDE_5DOCS = [
    ('tran_2019_01_test.txt', ['a', 'b', 'c', 'c']),
    ('tran_2019_02_test.txt', ['a', 'a', 'b', 'd']),
    ('tran_2019_03_test.txt', ['a', 'e', 'e', 'b']),
    ('tran_2020_01_test.txt', ['c', 'c', 'd', 'a']),
    ('tran_2020_02_test.txt', ['a', 'b', 'b', 'e']),
]

SIMPLE_CORPUS_ABCDEFG_7DOCS = [
    ('rand_1991_1.txt', ['b', 'd', 'a', 'c', 'e', 'b', 'a', 'd', 'b']),
    ('rand_1992_2.txt', ['b', 'f', 'e', 'e', 'f', 'e', 'a', 'a', 'b']),
    ('rand_1992_3.txt', ['a', 'e', 'f', 'b', 'e', 'a', 'b', 'f']),
    ('rand_1992_4.txt', ['e', 'a', 'a', 'b', 'g', 'f', 'g', 'b', 'c']),
    ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
    ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
    ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
]

SIMPLE_CORPUS_ABCDEFG_3DOCS = [
    ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
    ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
    ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
]


def generate_token2id(terms):
    token2id = defaultdict()
    token2id.default_factory = token2id.__len__
    for tokens in terms:
        for token in tokens:
            _ = token2id[token]
    return dict(token2id)


def very_simple_corpus(documents):

    reader = InMemoryReader(documents, filename_fields="year:_:1")
    corpus = TokenizedCorpus(reader=reader)
    return corpus


def random_corpus(n_docs: int = 5, vocabulary: str = 'abcdefg', min_length=4, max_length=10, years=None):

    def random_tokens():

        return [random.choice(vocabulary) for _ in range(0, random.choice(range(min_length, max_length)))]

    return [(f'rand_{random.choice(years or [0])}_{i}.txt', random_tokens()) for i in range(1, n_docs + 1)]


def test_co_occurrence_matrix_of_corpus_returns_correct_result():

    expected_token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    expected_matrix = np.matrix([[0, 6, 4, 3, 3], [0, 0, 2, 1, 4], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    v_corpus = CorpusVectorizer().fit_transform(corpus, vocabulary=corpus.token2id)

    term_term_matrix = v_corpus.co_occurrence_matrix()

    assert (term_term_matrix.todense() == expected_matrix).all()
    assert expected_token2id == v_corpus.token2id


def test_to_dataframe_has_same_values_as_coocurrence_matrix():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix = CorpusVectorizer().fit_transform(corpus, vocabulary=corpus.token2id).co_occurrence_matrix()

    df_coo = to_dataframe(
        term_term_matrix=term_term_matrix, id2token=corpus.id2token, documents=corpus.documents, n_count_threshold=1
    )

    assert df_coo.value.sum() == term_term_matrix.sum()
    assert 4 == int(df_coo[((df_coo.w1 == 'a') & (df_coo.w2 == 'c'))].value)
    assert 1 == int(df_coo[((df_coo.w1 == 'b') & (df_coo.w2 == 'd'))].value)


def test_to_coocurrence_matrix_yields_same_values_as_coocurrence_matrix():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix1 = CorpusVectorizer().fit_transform(corpus, vocabulary=corpus.token2id).co_occurrence_matrix()

    term_term_matrix2 = to_co_ocurrence_matrix(corpus)

    assert (term_term_matrix1 != term_term_matrix2).nnz == 0


def test_co_occurrence_given_windows_and_vocabulary_succeeds():

    windows_stream = [
        ['rand_1991_1.txt', 0, ['*', '*', 'b', 'd', 'a']],
        ['rand_1991_1.txt', 1, ['c', 'e', 'b', 'a', 'd']],
        ['rand_1991_1.txt', 2, ['a', 'd', 'b', '*', '*']],
        ['rand_1992_2.txt', 0, ['*', '*', 'b', 'f', 'e']],
        ['rand_1992_2.txt', 1, ['a', 'a', 'b', '*', '*']],
        ['rand_1992_3.txt', 0, ['e', 'f', 'b', 'e', 'a']],
        ['rand_1992_3.txt', 1, ['e', 'a', 'b', 'f', '*']],
        ['rand_1992_4.txt', 0, ['a', 'a', 'b', 'g', 'f']],
        ['rand_1992_4.txt', 1, ['f', 'g', 'b', 'c', '*']],
        ['rand_1991_5.txt', 0, ['*', 'c', 'b', 'c', 'e']],
        ['rand_1991_6.txt', 0, ['*', 'f', 'b', 'g', 'a']],
    ]
    vocabulary = generate_token2id([x[2] for x in windows_stream])

    windows_corpus = WindowsCorpus(windows_stream, vocabulary=vocabulary)

    v_corpus = CorpusVectorizer().fit_transform(windows_corpus, vocabulary=vocabulary)

    coo_matrix = v_corpus.co_occurrence_matrix()

    assert 10 == coo_matrix.todense()[vocabulary['b'], vocabulary['a']]
    assert 1 == coo_matrix.todense()[vocabulary['d'], vocabulary['c']]


def test_concept_co_occurrence_without_no_concept_and_threshold_succeeds():

    corpus = very_simple_corpus([
        ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
        ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
        ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
    ])
    expected_result = [('c', 'b', 2), ('b', 'g', 1), ('b', 'f', 1), ('g', 'f', 1)]

    coo_df = corpus_concept_co_occurrence(
        corpus, concepts={'b'}, no_concept=False, n_count_threshold=0, n_context_width=1
    )
    assert expected_result == dataframe_to_tuples(coo_df, ['w1', 'w2', 'value'])


def test_concept_co_occurrence_with_no_concept_succeeds():

    corpus = very_simple_corpus([
        ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
        ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
        ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
    ])
    expected_result = {('d', 'a', 1), ('b', 'a', 1)}

    coo_df = corpus_concept_co_occurrence(
        corpus, concepts={'g'}, no_concept=True, n_count_threshold=1, n_context_width=1
    )
    assert expected_result == set(dataframe_to_tuples(coo_df, ['w1', 'w2', 'value']))


def test_concept_co_occurrence_with_thresholdt_succeeds():

    corpus = very_simple_corpus([
        ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
        ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
        ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
    ])
    expected_result = {('g', 'a', 2)}

    coo_df = corpus_concept_co_occurrence(
        corpus, concepts={'g'}, no_concept=False, n_count_threshold=2, n_context_width=1
    )
    assert expected_result == set(dataframe_to_tuples(coo_df, ['w1', 'w2', 'value']))


def test_co_occurrence_using_cli_succeeds(tmpdir):

    output_filename = jj(tmpdir, 'test_co_occurrence_using_cli_succeeds.csv')
    options = dict(
        input_filename=TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME,
        output_filename=output_filename,
        concept={'jag'},
        context_width=2,
        partition_keys=['year'],
        pos_includes=None,
        pos_excludes='|MAD|MID|PAD|',
        lemmatize=True,
        to_lowercase=True,
        remove_stopwords=None,
        min_word_length=1,
        keep_symbols=True,
        keep_numerals=True,
        only_alphabetic=False,
        only_any_alphanumeric=False,
        filename_field=["year:_:1"],
    )

    cli_concept_co_occurrence(**options)

    assert os.path.isfile(output_filename)


@pytest.mark.parametrize("concept, n_count_threshold, n_context_width", [('fråga', 0, 2), ('fråga', 2, 2)])
def test_partitioned_corpus_concept_co_occurrence_succeeds(concept, n_count_threshold, n_context_width):

    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        tokenizer_opts=dict(filename_fields="year:_:1", ),
        pos_includes='|NN|VB|',
        lemmatize=False,
    )

    coo_df = partitioned_corpus_concept_co_occurrence(
        corpus,
        concepts={concept},
        no_concept=False,
        n_count_threshold=n_count_threshold,
        n_context_width=n_context_width,
        partition_keys='year',
    )

    assert coo_df is not None
    assert len(coo_df) > 0


@pytest.mark.skip("long running, used for bug fixes")
def test_co_occurrence_of_windowed_corpus_returns_correct_result4():

    concept = {'jag'}
    n_context_width = 2
    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.zip',
        tokenizer_opts=dict(filename_fields="year:_:1", ),
        pos_includes='|NN|VB|',
        lemmatize=False,
    )
    coo_df = partitioned_corpus_concept_co_occurrence(
        corpus,
        concepts=concept,
        no_concept=False,
        n_count_threshold=None,
        n_context_width=n_context_width,
        partition_keys='year',
    )

    assert coo_df is not None
    assert len(coo_df) > 0


def test_co_occurrence_bug_with_options_that_raises_an_exception(tmpdir):

    output_filename = jj(tmpdir, 'test_co_occurrence_bug_with_options_that_raises_an_exception.csv')
    options = {
        'input_filename': './tests/test_data/tranströmer_corpus_export.csv.zip',
        'output_filename': output_filename,
        'concept': ('jag', ),
        'context_width': 2,
        'partition_keys': ('year', ),
        'pos_includes': None,
        'pos_excludes': '|MAD|MID|PAD|',
        'lemmatize': True,
        'to_lowercase': True,
        'remove_stopwords': None,
        'min_word_length': 1,
        'keep_symbols': True,
        'keep_numerals': True,
        'only_alphabetic': False,
        'only_any_alphanumeric': False,
        'filename_field': ('year:_:1', ),
    }

    cli_concept_co_occurrence(**options)

    assert os.path.isfile(output_filename)
