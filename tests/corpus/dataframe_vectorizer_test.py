import pandas as pd

from penelope.co_occurrence import term_term_matrix_to_co_occurrences
from penelope.corpus import CorpusVectorizer, TokenizedCorpus, TokensTransformOpts
from penelope.corpus.readers import PandasCorpusReader


def create_test_dataframe():
    data = [(2000, 'A B C'), (2000, 'B C D'), (2001, 'C B'), (2003, 'A B F'), (2003, 'E B'), (2003, 'F E E')]
    df = pd.DataFrame(data, columns=['year', 'txt'])
    return df


def create_corpus():
    df = create_test_dataframe()
    reader = PandasCorpusReader(df)
    corpus = TokenizedCorpus(reader, transform_opts=TokensTransformOpts())
    return corpus


def test_corpus_token_stream():
    df = create_test_dataframe()
    reader = PandasCorpusReader(df)
    corpus = TokenizedCorpus(reader)
    result = [x for x in corpus]
    expected = [
        ('document_0.txt', ['A', 'B', 'C']),
        ('document_1.txt', ['B', 'C', 'D']),
        ('document_2.txt', ['C', 'B']),
        ('document_3.txt', ['A', 'B', 'F']),
        ('document_4.txt', ['E', 'B']),
        ('document_5.txt', ['F', 'E', 'E']),
    ]
    assert expected == result


def test_processed_corpus_token_stream():
    df = create_test_dataframe()
    reader = PandasCorpusReader(df)
    corpus = TokenizedCorpus(reader, transform_opts=TokensTransformOpts())
    result = [x for x in corpus]
    expected = [
        ('document_0.txt', ['A', 'B', 'C']),
        ('document_1.txt', ['B', 'C', 'D']),
        ('document_2.txt', ['C', 'B']),
        ('document_3.txt', ['A', 'B', 'F']),
        ('document_4.txt', ['E', 'B']),
        ('document_5.txt', ['F', 'E', 'E']),
    ]
    assert expected == result


def create_simple_test_corpus(transform_opts: TokensTransformOpts):
    data = [
        (2000, 'Detta är en mening med 14 token, 3 siffror och 2 symboler.'),
        (2000, 'Är det i denna mening en mening?'),
    ]
    df = pd.DataFrame(data, columns=['year', 'txt'])
    reader = PandasCorpusReader(df)
    corpus = TokenizedCorpus(reader, transform_opts=transform_opts)
    return corpus


def test_tokenized_document_where_symbols_are_filtered_out():
    corpus = create_simple_test_corpus(
        transform_opts=TokensTransformOpts(
            transforms={
                'remove-symbols': True,
                'only-any-alphanumeric': True,
            }
        )
    )
    result = [x for x in corpus]
    expected = [
        (
            'document_0.txt',
            ['Detta', 'är', 'en', 'mening', 'med', '14', 'token', '3', 'siffror', 'och', '2', 'symboler'],
        ),
        ('document_1.txt', ['Är', 'det', 'i', 'denna', 'mening', 'en', 'mening']),
    ]
    assert expected == result


def test_tokenized_document_where_symbols_and_numerals_are_filtered_out():
    corpus = create_simple_test_corpus(
        transform_opts=TokensTransformOpts(
            transforms={
                'remove_symbols': True,
                'only-any-alphanumeric': True,
                'remove-numerals': True,
            }
        )
    )
    result = [x for x in corpus]
    expected = [
        ('document_0.txt', ['Detta', 'är', 'en', 'mening', 'med', 'token', 'siffror', 'och', 'symboler']),
        ('document_1.txt', ['Är', 'det', 'i', 'denna', 'mening', 'en', 'mening']),
    ]
    assert expected == result


def test_tokenized_document_in_lowercase_where_symbols_and_numerals_and_one_letter_words_are_filtered_out():
    corpus = create_simple_test_corpus(
        transform_opts=TokensTransformOpts(
            transforms={
                'remove-symbols': True,
                'only-any-alphanumeric': True,
                'to-lower': True,
                'min-len': 2,
                'remove-numerals': True,
            }
        )
    )
    result = [x for x in corpus]
    expected = [
        ('document_0.txt', ['detta', 'är', 'en', 'mening', 'med', 'token', 'siffror', 'och', 'symboler']),
        ('document_1.txt', ['är', 'det', 'denna', 'mening', 'en', 'mening']),
    ]
    assert expected == result


def test_tokenized_document_in_lowercase_where_symbols_and_numerals_and_one_letter_words_and_stopwords_are_filtered_out():
    stopwords = {'är', 'en', 'med', 'och', 'det', 'detta', 'denna'}
    corpus = create_simple_test_corpus(
        transform_opts=TokensTransformOpts(
            transforms={
                'remove-symbols': True,
                'remove-numerals': True,
                'only-any-alphanumeric': True,
                'to-lower': True,
                'min-chars': 2,
            },
            extra_stopwords=stopwords,
        )
    )
    result = [x for x in corpus]
    expected = [
        ('document_0.txt', ['mening', 'token', 'siffror', 'symboler']),
        ('document_1.txt', ['mening', 'mening']),
    ]
    assert expected == result


def test_fit_transform_gives_document_term_matrix():
    # Arrange
    reader = PandasCorpusReader(create_test_dataframe())
    corpus = TokenizedCorpus(
        reader,
        transform_opts=TokensTransformOpts(
            transforms={'remove-numerals': True},
        ),
    )
    v_corpus = CorpusVectorizer().fit_transform(corpus)

    term_term_matrix = v_corpus.co_occurrence_matrix()
    token2id = v_corpus.token2id

    assert 2 == term_term_matrix.todense()[token2id['A'], token2id['B']]
    assert 0 == term_term_matrix.todense()[token2id['C'], token2id['F']]


def test_to_dataframe_of_term_matrix_gives_expected_result():
    # Arrange
    reader = PandasCorpusReader(create_test_dataframe())
    corpus = TokenizedCorpus(reader, transform_opts=TokensTransformOpts(transforms={'remove-numerals': True}))

    term_term_matrix = CorpusVectorizer().fit_transform(corpus, already_tokenized=True).co_occurrence_matrix()

    # Act
    id2w = corpus.id2token.get
    co_occurrences = term_term_matrix_to_co_occurrences(term_term_matrix, threshold_count=1, ignore_ids=set())
    co_occurrences['w1'] = co_occurrences.w1_id.apply(id2w)
    co_occurrences['w2'] = co_occurrences.w2_id.apply(id2w)

    # Assert
    assert 2 == int(co_occurrences[((co_occurrences.w1 == 'A') & (co_occurrences.w2 == 'B'))].value)
    assert 0 == len(co_occurrences[((co_occurrences.w1 == 'C') & (co_occurrences.w2 == 'F'))])


def test_tokenized_document_token_counts_is_empty_if_enumerable_not_exhausted():
    corpus = create_simple_test_corpus(
        transform_opts=TokensTransformOpts(
            transforms={
                'remove-symbols': True,
                'only-any-alphanumeric': True,
                'to-lower': True,
            }
        )
    )
    assert 'n_raw_tokens' not in corpus.document_index.columns
    assert 'n_tokens' not in corpus.document_index.columns


def test_tokenized_document_token_counts_is_not_empty_if_enumerable_is_exhausted():
    # Note: Symbols are always removed by reader - hence "keep_symbols" filter has no effect
    corpus = create_simple_test_corpus(
        transform_opts=TokensTransformOpts(
            transforms={
                'remove-symbols': True,
                'only-any-alphanumeric': True,
                'to-lower': True,
            }
        )
    )
    for _ in corpus:
        pass
    assert 'n_raw_tokens' in corpus.document_index.columns
    assert 'n_tokens' in corpus.document_index.columns
