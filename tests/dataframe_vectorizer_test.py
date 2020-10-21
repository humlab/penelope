import unittest

import pandas as pd

import penelope.corpus.readers as readers
import penelope.corpus.tokenized_corpus as corpora
from penelope.cooccurrence.term_term_matrix import to_dataframe
from penelope.corpus import vectorizer as corpus_vectorizer

DEFAULT_TOKENS_TRANSFORM_OPTS = dict(
    only_any_alphanumeric=False,
    to_lower=False,
    remove_accents=False,
    min_len=1,
    max_len=None,
    keep_numerals=False
)


class Test_DataFrameVectorize(unittest.TestCase):

    def setUp(self):
        pass

    def create_test_dataframe(self):
        data = [(2000, 'A B C'), (2000, 'B C D'), (2001, 'C B'), (2003, 'A B F'), (2003, 'E B'), (2003, 'F E E')]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        return df

    def create_corpus(self):
        df = self.create_test_dataframe()
        reader = readers.DataFrameTextTokenizer(df)
        corpus = corpora.TokenizedCorpus(reader, **DEFAULT_TOKENS_TRANSFORM_OPTS)
        return corpus

    def test_corpus_token_stream(self):
        df = self.create_test_dataframe()
        reader = readers.DataFrameTextTokenizer(df)
        corpus = corpora.TokenizedCorpus(reader)
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
        reader = readers.DataFrameTextTokenizer(df)
        corpus = corpora.TokenizedCorpus(reader, **DEFAULT_TOKENS_TRANSFORM_OPTS)
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

    def create_simple_test_corpus(self, **kwargs):
        data = [
            (2000, 'Detta är en mening med 14 token, 3 siffror och 2 symboler.'),
            (2000, 'Är det i denna mening en mening?'),
        ]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        reader = readers.DataFrameTextTokenizer(df)
        corpus = corpora.TokenizedCorpus(reader, **kwargs)
        return corpus

    def test_tokenized_document_where_symbols_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(
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
        result = [x for x in corpus]
        expected = [
            ('0', ['Detta', 'är', 'en', 'mening', 'med', '14', 'token', '3', 'siffror', 'och', '2', 'symboler']),
            ('1', ['Är', 'det', 'i', 'denna', 'mening', 'en', 'mening']),
        ]
        self.assertEqual(expected, result)

    def test_tokenized_document_where_symbols_and_numerals_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(
            keep_symbols=False,
            only_any_alphanumeric=True,
            to_lower=False,
            remove_accents=False,
            min_len=0,
            max_len=None,
            keep_numerals=False,
            stopwords=None,
        )
        result = [x for x in corpus]
        expected = [
            ('0', ['Detta', 'är', 'en', 'mening', 'med', 'token', 'siffror', 'och', 'symboler']),
            ('1', ['Är', 'det', 'i', 'denna', 'mening', 'en', 'mening']),
        ]
        self.assertEqual(expected, result)

    def test_tokenized_document_in_lowercase_where_symbols_and_numerals_and_one_letter_words_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(
            keep_symbols=False,
            only_any_alphanumeric=True,
            to_lower=True,
            remove_accents=False,
            min_len=2,
            max_len=None,
            keep_numerals=False,
            stopwords=None,
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
            keep_symbols=False,
            only_any_alphanumeric=True,
            to_lower=True,
            remove_accents=False,
            min_len=2,
            max_len=None,
            keep_numerals=False,
            stopwords=stopwords,
        )
        result = [x for x in corpus]
        expected = [
            ('0', ['mening', 'token', 'siffror', 'symboler']),
            ('1', ['mening', 'mening']),
        ]
        self.assertEqual(expected, result)

    def test_fit_transform_gives_document_term_matrix(self):
        # Arrange
        reader = readers.DataFrameTextTokenizer(
            self.create_test_dataframe()
        )
        corpus = corpora.TokenizedCorpus(
            reader,
            only_any_alphanumeric=False,
            to_lower=False,
            remove_accents=False,
            min_len=1,
            max_len=None,
            keep_numerals=False
        )
        v_corpus = corpus_vectorizer\
            .CorpusVectorizer()\
            .fit_transform(corpus)

        term_term_matrix = v_corpus.cooccurrence_matrix()
        token2id = v_corpus.token2id

        assert 2 == term_term_matrix.todense()[token2id['A'], token2id['B']]
        assert 0 == term_term_matrix.todense()[token2id['C'], token2id['F']]

    def test_to_dataframe_of_term_matrix_gives_expected_result(self):

        # Arrange
        reader = readers.DataFrameTextTokenizer(
            self.create_test_dataframe()
        )
        corpus = corpora.TokenizedCorpus(
            reader,
            only_any_alphanumeric=False,
            to_lower=False,
            remove_accents=False,
            min_len=1,
            max_len=None,
            keep_numerals=False
        )

        term_term_matrix = corpus_vectorizer\
            .CorpusVectorizer()\
            .fit_transform(corpus)\
            .cooccurrence_matrix()

        # Act
        coo_df = to_dataframe(term_term_matrix, corpus.id2token, corpus.documents)

        # Assert
        assert 2 == int(coo_df[((coo_df.w1 == 'A') & (coo_df.w2 == 'B'))].value)
        assert 0 == len(coo_df[((coo_df.w1 == 'C') & (coo_df.w2 == 'F'))])

    def test_tokenized_document_token_counts_is_empty_if_enumerable_not_exhausted(self):
        corpus = self.create_simple_test_corpus(
            keep_symbols=False,
            only_any_alphanumeric=True,
            to_lower=True,
            remove_accents=False,
            min_len=0,
            max_len=None,
            keep_numerals=True,
            stopwords=None,
        )
        self.assertTrue('n_raw_tokens' not in corpus.documents.columns)
        self.assertTrue('n_tokens' not in corpus.documents.columns)

    def test_tokenized_document_token_counts_is_not_empty_if_enumerable_is_exhausted(self):
        # Note: Symbols are always removed by reader - hence "keep_symbols" filter has no effect
        corpus = self.create_simple_test_corpus(
            keep_symbols=False,
            only_any_alphanumeric=True,
            to_lower=True,
            remove_accents=False,
            min_len=0,
            max_len=None,
            keep_numerals=True,
            stopwords=None,
        )
        for _ in corpus:
            pass
        self.assertTrue('n_raw_tokens' in corpus.documents.columns)
        self.assertTrue('n_tokens' in corpus.documents.columns)
