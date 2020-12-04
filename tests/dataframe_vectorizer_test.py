import unittest

import pandas as pd
from penelope.co_occurrence.term_term_matrix import to_dataframe
from penelope.corpus import CorpusVectorizer, TokenizedCorpus, TokensTransformOpts
from penelope.corpus.readers import DataFrameTextTokenizer


class Test_DataFrameVectorize(unittest.TestCase):
    def setUp(self):
        pass

    def create_test_dataframe(self):
        data = [(2000, 'A B C'), (2000, 'B C D'), (2001, 'C B'), (2003, 'A B F'), (2003, 'E B'), (2003, 'F E E')]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        return df

    def create_corpus(self):
        df = self.create_test_dataframe()
        reader = DataFrameTextTokenizer(df)
        corpus = TokenizedCorpus(reader, tokens_transform_opts=TokensTransformOpts())
        return corpus

    def test_corpus_token_stream(self):
        df = self.create_test_dataframe()
        reader = DataFrameTextTokenizer(df)
        corpus = TokenizedCorpus(reader)
        result = [x for x in corpus]
        expected = [
            ('0', ['A', 'B', 'C']),
            ('1', ['B', 'C', 'D']),
            ('2', ['C', 'B']),
            ('3', ['A', 'B', 'F']),
            ('4', ['E', 'B']),
            ('5', ['F', 'E', 'E']),
        ]
        self.assertEqual(expected, result)

    def test_processed_corpus_token_stream(self):
        df = self.create_test_dataframe()
        reader = DataFrameTextTokenizer(df)
        corpus = TokenizedCorpus(reader, tokens_transform_opts=TokensTransformOpts())
        result = [x for x in corpus]
        expected = [
            ('0', ['A', 'B', 'C']),
            ('1', ['B', 'C', 'D']),
            ('2', ['C', 'B']),
            ('3', ['A', 'B', 'F']),
            ('4', ['E', 'B']),
            ('5', ['F', 'E', 'E']),
        ]
        self.assertEqual(expected, result)

    def create_simple_test_corpus(self, tokens_transform_opts: TokensTransformOpts):
        data = [
            (2000, 'Detta är en mening med 14 token, 3 siffror och 2 symboler.'),
            (2000, 'Är det i denna mening en mening?'),
        ]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        reader = DataFrameTextTokenizer(df)
        corpus = TokenizedCorpus(reader, tokens_transform_opts=tokens_transform_opts)
        return corpus

    def test_tokenized_document_where_symbols_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(
            tokens_transform_opts=TokensTransformOpts(
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
            ('0', ['Detta', 'är', 'en', 'mening', 'med', '14', 'token', '3', 'siffror', 'och', '2', 'symboler']),
            ('1', ['Är', 'det', 'i', 'denna', 'mening', 'en', 'mening']),
        ]
        self.assertEqual(expected, result)

    def test_tokenized_document_where_symbols_and_numerals_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(
            tokens_transform_opts=TokensTransformOpts(
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
            ('0', ['Detta', 'är', 'en', 'mening', 'med', 'token', 'siffror', 'och', 'symboler']),
            ('1', ['Är', 'det', 'i', 'denna', 'mening', 'en', 'mening']),
        ]
        self.assertEqual(expected, result)

    def test_tokenized_document_in_lowercase_where_symbols_and_numerals_and_one_letter_words_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(
            tokens_transform_opts=TokensTransformOpts(
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
            ('0', ['detta', 'är', 'en', 'mening', 'med', 'token', 'siffror', 'och', 'symboler']),
            ('1', ['är', 'det', 'denna', 'mening', 'en', 'mening']),
        ]
        self.assertEqual(expected, result)

    def test_tokenized_document_in_lowercase_where_symbols_and_numerals_and_one_letter_words_and_stopwords_are_filtered_out(
        self,
    ):
        stopwords = {'är', 'en', 'med', 'och', 'det', 'detta', 'denna'}
        corpus = self.create_simple_test_corpus(
            tokens_transform_opts=TokensTransformOpts(
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
            ('0', ['mening', 'token', 'siffror', 'symboler']),
            ('1', ['mening', 'mening']),
        ]
        self.assertEqual(expected, result)

    def test_fit_transform_gives_document_term_matrix(self):
        # Arrange
        reader = DataFrameTextTokenizer(self.create_test_dataframe())
        corpus = TokenizedCorpus(
            reader,
            tokens_transform_opts=TokensTransformOpts(
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
        reader = DataFrameTextTokenizer(self.create_test_dataframe())
        corpus = TokenizedCorpus(
            reader,
            tokens_transform_opts=TokensTransformOpts(
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
        coo_df = to_dataframe(term_term_matrix, corpus.id2token, corpus.documents)

        # Assert
        assert 2 == int(coo_df[((coo_df.w1 == 'A') & (coo_df.w2 == 'B'))].value)
        assert 0 == len(coo_df[((coo_df.w1 == 'C') & (coo_df.w2 == 'F'))])

    def test_tokenized_document_token_counts_is_empty_if_enumerable_not_exhausted(self):
        corpus = self.create_simple_test_corpus(
            tokens_transform_opts=TokensTransformOpts(
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
        self.assertTrue('n_raw_tokens' not in corpus.documents.columns)
        self.assertTrue('n_tokens' not in corpus.documents.columns)

    def test_tokenized_document_token_counts_is_not_empty_if_enumerable_is_exhausted(self):
        # Note: Symbols are always removed by reader - hence "keep_symbols" filter has no effect
        corpus = self.create_simple_test_corpus(
            tokens_transform_opts=TokensTransformOpts(
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
        self.assertTrue('n_raw_tokens' in corpus.documents.columns)
        self.assertTrue('n_tokens' in corpus.documents.columns)
