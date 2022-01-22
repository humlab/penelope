import unittest

import pandas as pd

from penelope.co_occurrence import term_term_matrix_to_co_occurrences
from penelope.corpus import CorpusVectorizer, TokenizedCorpus, TokensTransformOpts
from penelope.corpus.readers import PandasCorpusReader


class Test_DataFrameVectorize(unittest.TestCase):
    def setUp(self):
        pass

    def create_test_dataframe(self):
        data = [(2000, 'A B C'), (2000, 'B C D'), (2001, 'C B'), (2003, 'A B F'), (2003, 'E B'), (2003, 'F E E')]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        return df

    def create_corpus(self):
        df = self.create_test_dataframe()
        reader = PandasCorpusReader(df)
        corpus = TokenizedCorpus(reader, transform_opts=TokensTransformOpts())
        return corpus

    def test_corpus_token_stream(self):
        df = self.create_test_dataframe()
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
        self.assertEqual(expected, result)

    def test_processed_corpus_token_stream(self):
        df = self.create_test_dataframe()
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
        self.assertEqual(expected, result)

    def create_simple_test_corpus(self, transform_opts: TokensTransformOpts):
        data = [
            (2000, 'Detta är en mening med 14 token, 3 siffror och 2 symboler.'),
            (2000, 'Är det i denna mening en mening?'),
        ]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        reader = PandasCorpusReader(df)
        corpus = TokenizedCorpus(reader, transform_opts=transform_opts)
        return corpus

    def test_tokenized_document_where_symbols_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(
            transform_opts=TokensTransformOpts(
                keep_symbols=False,
                only_any_alphanumeric=True,
                to_lower=False,
                remove_accents=False,
                min_len=0,
                max_len=None,
                keep_numerals=True,
                stopwords=None,
                only_alphabetic=False,
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
        self.assertEqual(expected, result)

    def test_tokenized_document_where_symbols_and_numerals_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(
            transform_opts=TokensTransformOpts(
                keep_symbols=False,
                only_any_alphanumeric=True,
                to_lower=False,
                remove_accents=False,
                min_len=0,
                max_len=None,
                keep_numerals=False,
                stopwords=None,
            )
        )
        result = [x for x in corpus]
        expected = [
            ('document_0.txt', ['Detta', 'är', 'en', 'mening', 'med', 'token', 'siffror', 'och', 'symboler']),
            ('document_1.txt', ['Är', 'det', 'i', 'denna', 'mening', 'en', 'mening']),
        ]
        self.assertEqual(expected, result)

    def test_tokenized_document_in_lowercase_where_symbols_and_numerals_and_one_letter_words_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(
            transform_opts=TokensTransformOpts(
                keep_symbols=False,
                only_any_alphanumeric=True,
                to_lower=True,
                remove_accents=False,
                min_len=2,
                max_len=None,
                keep_numerals=False,
                stopwords=None,
            )
        )
        result = [x for x in corpus]
        expected = [
            ('document_0.txt', ['detta', 'är', 'en', 'mening', 'med', 'token', 'siffror', 'och', 'symboler']),
            ('document_1.txt', ['är', 'det', 'denna', 'mening', 'en', 'mening']),
        ]
        self.assertEqual(expected, result)

    def test_tokenized_document_in_lowercase_where_symbols_and_numerals_and_one_letter_words_and_stopwords_are_filtered_out(
        self,
    ):
        stopwords = {'är', 'en', 'med', 'och', 'det', 'detta', 'denna'}
        corpus = self.create_simple_test_corpus(
            transform_opts=TokensTransformOpts(
                keep_symbols=False,
                only_any_alphanumeric=True,
                to_lower=True,
                remove_accents=False,
                min_len=2,
                max_len=None,
                keep_numerals=False,
                stopwords=stopwords,
            )
        )
        result = [x for x in corpus]
        expected = [
            ('document_0.txt', ['mening', 'token', 'siffror', 'symboler']),
            ('document_1.txt', ['mening', 'mening']),
        ]
        self.assertEqual(expected, result)

    def test_fit_transform_gives_document_term_matrix(self):
        # Arrange
        reader = PandasCorpusReader(self.create_test_dataframe())
        corpus = TokenizedCorpus(
            reader,
            transform_opts=TokensTransformOpts(
                only_any_alphanumeric=False,
                to_lower=False,
                remove_accents=False,
                min_len=1,
                max_len=None,
                keep_numerals=False,
            ),
        )
        v_corpus = CorpusVectorizer().fit_transform(corpus)

        term_term_matrix = v_corpus.co_occurrence_matrix()
        token2id = v_corpus.token2id

        assert 2 == term_term_matrix.todense()[token2id['A'], token2id['B']]
        assert 0 == term_term_matrix.todense()[token2id['C'], token2id['F']]

    def test_to_dataframe_of_term_matrix_gives_expected_result(self):

        # Arrange
        reader = PandasCorpusReader(self.create_test_dataframe())
        corpus = TokenizedCorpus(
            reader,
            # Pre-compute transform options:
            transform_opts=TokensTransformOpts(
                only_any_alphanumeric=False,
                to_lower=False,
                remove_accents=False,
                min_len=1,
                max_len=None,
                keep_numerals=False,
            ),
        )

        term_term_matrix = CorpusVectorizer().fit_transform(corpus, already_tokenized=True).co_occurrence_matrix()

        # Act
        id2w = corpus.id2token.get
        co_occurrences = term_term_matrix_to_co_occurrences(term_term_matrix, threshold_count=1, ignore_ids=set())
        co_occurrences['w1'] = co_occurrences.w1_id.apply(id2w)
        co_occurrences['w2'] = co_occurrences.w2_id.apply(id2w)

        # Assert
        assert 2 == int(co_occurrences[((co_occurrences.w1 == 'A') & (co_occurrences.w2 == 'B'))].value)
        assert 0 == len(co_occurrences[((co_occurrences.w1 == 'C') & (co_occurrences.w2 == 'F'))])

    def test_tokenized_document_token_counts_is_empty_if_enumerable_not_exhausted(self):
        corpus = self.create_simple_test_corpus(
            transform_opts=TokensTransformOpts(
                keep_symbols=False,
                only_any_alphanumeric=True,
                to_lower=True,
                remove_accents=False,
                min_len=0,
                max_len=None,
                keep_numerals=True,
                stopwords=None,
            )
        )
        self.assertTrue('n_raw_tokens' not in corpus.document_index.columns)
        self.assertTrue('n_tokens' not in corpus.document_index.columns)

    def test_tokenized_document_token_counts_is_not_empty_if_enumerable_is_exhausted(self):
        # Note: Symbols are always removed by reader - hence "keep_symbols" filter has no effect
        corpus = self.create_simple_test_corpus(
            transform_opts=TokensTransformOpts(
                keep_symbols=False,
                only_any_alphanumeric=True,
                to_lower=True,
                remove_accents=False,
                min_len=0,
                max_len=None,
                keep_numerals=True,
                stopwords=None,
            )
        )
        for _ in corpus:
            pass
        self.assertTrue('n_raw_tokens' in corpus.document_index.columns)
        self.assertTrue('n_tokens' in corpus.document_index.columns)
