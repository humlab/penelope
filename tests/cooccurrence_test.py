import os
import random

import pytest

from penelope.cooccurrence import (WindowsCorpus, cooccurrence_by_partition,
                                   corpus_concept_windows,
                                   to_coocurrence_matrix, to_dataframe)
from penelope.corpus import CorpusVectorizer, TokenizedCorpus
from penelope.corpus.readers import InMemoryReader
from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus
from penelope.scripts.concept_cooccurrence import \
    compute_and_store_cooccerrence

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


def very_simple_corpus(documents):

    reader = InMemoryReader(documents, filename_fields="year:_:1")
    corpus = TokenizedCorpus(reader=reader)
    return corpus


def random_corpus(n_docs: int = 5, vocabulary: str = 'abcdefg', min_length=4, max_length=10, years=None):

    def random_tokens():

        return [random.choice(vocabulary) for _ in range(0, random.choice(range(min_length, max_length)))]

    return [(f'rand_{random.choice(years or [0])}_{i}.txt', random_tokens()) for i in range(1, n_docs + 1)]


def test_cooccurrence_matrix_of_corpus_returns_correct_result():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix = CorpusVectorizer().fit_transform(corpus, vocabulary=corpus.token2id).cooccurrence_matrix()

    assert term_term_matrix is not None
    assert term_term_matrix.shape == (len(corpus.token2id), len(corpus.token2id))
    assert term_term_matrix.sum() == 25
    assert term_term_matrix.todense()[corpus.token2id['a'], corpus.token2id['c']] == 4
    assert term_term_matrix.todense()[corpus.token2id['b'], corpus.token2id['d']] == 1


def test_to_dataframe_has_same_values_as_coocurrence_matrix():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix = CorpusVectorizer().fit_transform(corpus, vocabulary=corpus.token2id).cooccurrence_matrix()

    df_coo = to_dataframe(
        term_term_matrix=term_term_matrix, id2token=corpus.id2token, documents=corpus.documents, min_count=1
    )

    assert df_coo.value.sum() == term_term_matrix.sum()
    assert 4 == int(df_coo[((df_coo.w1 == 'a') & (df_coo.w2 == 'c'))].value)
    assert 1 == int(df_coo[((df_coo.w1 == 'b') & (df_coo.w2 == 'd'))].value)


def test_to_coocurrence_matrix_yields_same_values_as_coocurrence_matrix():

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix1 = CorpusVectorizer().fit_transform(corpus, vocabulary=corpus.token2id).cooccurrence_matrix()

    term_term_matrix2 = to_coocurrence_matrix(corpus)

    assert (term_term_matrix1 != term_term_matrix2).nnz == 0


def test_cooccurrence_of_windowed_corpus_returns_correct_result():

    # corpus = SparvTokenizedCsvCorpus(SPARV_ZIPPED_CSV_EXPORT_FILENAME, pos_includes='|NN|VB|', lemmatize=False)
    # vocabulary = corpus.token2id
    documents = SIMPLE_CORPUS_ABCDEFG_7DOCS

    expected_windows = [
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

    vocabulary = {chr(ord('a') + i): i for i in range(0, ord('f') - ord('a') + 1)}
    concept = {'b'}
    windows = [w for w in corpus_concept_windows(documents, concept, n_context_width=2, pad='*')]

    assert expected_windows == windows

    windows_corpus = WindowsCorpus(windows, vocabulary=vocabulary)

    v_corpus = CorpusVectorizer().fit_transform(windows_corpus)

    coo_matrix = v_corpus.cooccurrence_matrix()

    assert coo_matrix is not None

    # TODO Add more result asserts

    assert 10 == coo_matrix.todense()[vocabulary['a'], vocabulary['b']]
    assert 1 == coo_matrix.todense()[vocabulary['c'], vocabulary['d']]


def test_cooccurrence_of_windowed_corpus_returns_correct_result2():

    concept = {'är'}
    n_lr_tokens = 2
    corpus = SparvTokenizedCsvCorpus(
        TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME,
        tokenizer_opts=dict(filename_fields="year:_:1", ),
        pos_includes='|NN|VB|',
        lemmatize=False,
    )
    coo_df = cooccurrence_by_partition(corpus, concept, n_lr_tokens)

    assert coo_df is not None


def test_cooccurrence_using_cli_succeeds():

    output_filename = './tests/output/test_cooccurrence_using_cli_succeeds.csv'
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

    compute_and_store_cooccerrence(**options)

    assert os.path.isfile(output_filename)

@pytest.skip("long running, used for bug fixes")
def test_cooccurrence_of_windowed_corpus_returns_correct_result3():

    concept = {'jag'}
    n_lr_tokens = 2
    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.2files.zip',
        tokenizer_opts=dict(filename_fields="year:_:1", ),
        pos_includes='|NN|VB|',
        lemmatize=False,
    )
    coo_df = cooccurrence_by_partition(corpus, concept, n_lr_tokens)

    assert coo_df is not None


@pytest.skip("long running, used for bug fixes")
def test_cooccurrence_of_windowed_corpus_returns_correct_result4():

    concept = {'jag'}
    n_lr_tokens = 2
    corpus = SparvTokenizedCsvCorpus(
        './tests/test_data/riksdagens-protokoll.1920-2019.test.zip',
        tokenizer_opts=dict(filename_fields="year:_:1", ),
        pos_includes='|NN|VB|',
        lemmatize=False,
    )
    coo_df = cooccurrence_by_partition(corpus, concept, n_lr_tokens)

    assert coo_df is not None
    assert len(coo_df) > 0
