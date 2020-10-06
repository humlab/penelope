import unittest
import pandas as pd
import numpy as np
import scipy

from penelope.corpus import vectorizer as corpus_vectorizer
import penelope.corpus.tokenized_corpus as corpora
import penelope.corpus.readers as readers


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
        kwargs = dict(
            only_any_alphanumeric=False,
            to_lower=False,
            remove_accents=False,
            min_len=0,
            max_len=None,
            keep_numerals=False,
        )
        corpus = corpora.TokenizedCorpus(reader, **kwargs)
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
        kwargs = dict(
            only_any_alphanumeric=False,
            to_lower=False,
            remove_accents=False,
            min_len=0,
            max_len=None,
            keep_numerals=False,
        )
        corpus = corpora.TokenizedCorpus(reader, **kwargs)
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
        reader = readers.DataFrameTextTokenizer(self.create_test_dataframe())
        kwargs = dict(to_lower=False, remove_accents=False, min_len=1, max_len=None, keep_numerals=False)
        corpus = corpora.TokenizedCorpus(reader, only_any_alphanumeric=False, **kwargs)
        vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
        v_corpus = vectorizer.fit_transform(corpus)
        expected = np.asarray(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 2, 1],
            ]
        )
        self.assertTrue((expected == v_corpus.bag_term_matrix).all())
        results = v_corpus.token2id
        expected = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 5, 'E': 4}
        self.assertEqual(expected, results)

    def test_AxAt_of_document_term_matrix_gives_term_term_matrix(self):

        # Arrange
        reader = readers.DataFrameTextTokenizer(self.create_test_dataframe())
        kwargs = dict(to_lower=False, remove_accents=False, min_len=1, max_len=None, keep_numerals=False)
        corpus = corpora.TokenizedCorpus(reader, only_any_alphanumeric=False, **kwargs)
        vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
        v_corpus = vectorizer.fit_transform(corpus)

        # Act
        term_term_matrix = np.dot(v_corpus.bag_term_matrix.T, v_corpus.bag_term_matrix)

        # Assert
        expected = np.asarray(
            [
                [2, 2, 1, 0, 0, 1],
                [2, 5, 3, 1, 1, 1],
                [1, 3, 3, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 5, 2],
                [1, 1, 0, 0, 2, 2],
            ]
        )
        self.assertTrue((expected == term_term_matrix).all())

        term_term_matrix = scipy.sparse.triu(term_term_matrix, 1)
        expected = np.asarray(
            [
                [0, 2, 1, 0, 0, 1],
                [0, 0, 3, 1, 1, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        # print(term_term_matrix.todense())
        # print(term_term_matrix)
        coo = term_term_matrix
        id2token = {i: t for t, i in v_corpus.token2id.items()}
        cdf = (
            pd.DataFrame({'w1_id': coo.row, 'w2_id': coo.col, 'value': coo.data})[['w1_id', 'w2_id', 'value']]
            .sort_values(['w1_id', 'w2_id'])
            .reset_index(drop=True)
        )
        cdf['w1'] = cdf.w1_id.apply(lambda x: id2token[x])
        cdf['w2'] = cdf.w2_id.apply(lambda x: id2token[x])
        print(cdf[['w1', 'w2', 'value']])

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
