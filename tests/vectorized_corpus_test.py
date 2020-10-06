import os
import unittest

import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import CountVectorizer

import penelope.corpus.tokenized_corpus as corpora
from penelope.corpus import corpus_vectorizer, vectorized_corpus
from tests.utils import create_text_tokenizer


def flatten(l):
    return [x for ws in l for x in ws]


TEMP_OUTPUT_FOLDER = "./penelope/tests/output"

# pylint: disable=too-many-public-methods


class Test_VectorizedCorpus(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEMP_OUTPUT_FOLDER, exist_ok=True)

    def create_reader(self):
        filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_tokenizer(filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True)
        return reader

    def create_corpus(self):
        reader = self.create_reader()
        kwargs = dict(
            only_any_alphanumeric=True,
            to_lower=True,
            remove_accents=False,
            min_len=2,
            max_len=None,
            keep_numerals=False,
        )
        corpus = corpora.TokenizedCorpus(reader, **kwargs)
        return corpus

    def create_vectorized_corpus(self):
        bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
        token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        df = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
        v_corpus = vectorized_corpus.VectorizedCorpus(bag_term_matrix, token2id, df)
        return v_corpus

    def test_bag_term_matrix_to_bag_term_docs(self):

        v_corpus = self.create_vectorized_corpus()

        doc_ids = (
            0,
            1,
        )
        expected = [['a', 'a', 'b', 'c', 'c', 'c', 'c', 'd'], ['a', 'a', 'b', 'b', 'c', 'c', 'c']]
        docs = v_corpus.to_bag_of_terms(doc_ids)
        assert expected == ([list(d) for d in docs])

        expected = [
            ['a', 'a', 'b', 'c', 'c', 'c', 'c', 'd'],
            ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
            ['a', 'a', 'b', 'b', 'b', 'c', 'c'],
            ['a', 'a', 'b', 'b', 'b', 'b', 'c', 'd'],
            ['a', 'a', 'c', 'd'],
        ]
        docs = v_corpus.to_bag_of_terms()
        assert expected == ([list(d) for d in docs])

    def test_load_of_uncompressed_corpus(self):

        # Arrange
        corpus = self.create_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        dumped_v_corpus = vectorizer.fit_transform(corpus)

        dumped_v_corpus.dump('dump_test', folder=TEMP_OUTPUT_FOLDER, compressed=False)

        # Act
        loaded_v_corpus = vectorized_corpus.VectorizedCorpus.load('dump_test', folder=TEMP_OUTPUT_FOLDER)

        # Assert
        self.assertEqual(dumped_v_corpus.word_counts, loaded_v_corpus.word_counts)
        self.assertEqual(dumped_v_corpus.document_index.to_dict(), loaded_v_corpus.document_index.to_dict())
        self.assertEqual(dumped_v_corpus.token2id, loaded_v_corpus.token2id)
        # self.assertEqual(dumped_v_corpus.X, loaded_v_corpus.X)

    def test_load_of_compressed_corpus(self):

        # Arrange
        corpus = self.create_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        dumped_v_corpus = vectorizer.fit_transform(corpus)

        dumped_v_corpus.dump('dump_test', folder=TEMP_OUTPUT_FOLDER, compressed=True)

        # Act
        loaded_v_corpus = vectorized_corpus.VectorizedCorpus.load('dump_test', folder=TEMP_OUTPUT_FOLDER)

        # Assert
        self.assertEqual(dumped_v_corpus.word_counts, loaded_v_corpus.word_counts)
        self.assertEqual(dumped_v_corpus.document_index.to_dict(), loaded_v_corpus.document_index.to_dict())
        self.assertEqual(dumped_v_corpus.token2id, loaded_v_corpus.token2id)
        # self.assertEqual(dumped_v_corpus.X, loaded_v_corpus.X)

    def test_group_by_year_aggregates_bag_term_matrix_to_year_term_matrix(self):
        v_corpus = self.create_vectorized_corpus()
        c_data = v_corpus.group_by_year()
        expected_ytm = [[4, 3, 7, 1], [6, 7, 4, 2]]
        self.assertTrue(np.allclose(expected_ytm, c_data.bag_term_matrix.todense()))

    def test_group_by_year2_sum_bag_term_matrix_to_year_term_matrix(self):
        v_corpus = self.create_vectorized_corpus()
        c_data = v_corpus.group_by_year2(aggregate_function='sum')
        expected_ytm = [[4, 3, 7, 1], [6, 7, 4, 2]]
        self.assertTrue(np.allclose(expected_ytm, c_data.bag_term_matrix.todense()))
        self.assertEqual(v_corpus.data.dtype, c_data.data.dtype)

    def test_group_by_year2_mean_bag_term_matrix_to_year_term_matrix(self):
        v_corpus = self.create_vectorized_corpus()
        c_data = v_corpus.group_by_year2(aggregate_function='mean', dtype=np.float)
        expected_ytm = [np.array([4.0, 3.0, 7.0, 1.0]) / 2.0, np.array([6.0, 7.0, 4.0, 2.0]) / 3.0]
        self.assertTrue(np.allclose(expected_ytm, c_data.bag_term_matrix.todense()))

    def test_collapse_to_category_aggregates_bag_term_matrix_to_category_term_matrix(self):
        """ A more generic version of group_by_year (not used for now) """
        v_corpus = self.create_vectorized_corpus()
        Y, _ = v_corpus.collapse_by_category('year')
        expected_ytm = [[4, 3, 7, 1], [6, 7, 4, 2]]
        self.assertTrue(np.allclose(expected_ytm, Y))

    def test_collapse_to_category_sums_bag_term_matrix_to_category_term_matrix(self):
        """ A more generic version of group_by_year (not used for now) """
        v_corpus = self.create_vectorized_corpus()
        Y, _ = v_corpus.collapse_by_category('year', aggregate_function='sum')
        expected_ytm = [[4, 3, 7, 1], [6, 7, 4, 2]]
        self.assertTrue(np.allclose(expected_ytm, Y))

    def test_collapse_to_category_means_bag_term_matrix_to_category_term_matrix(self):
        """ A more generic version of group_by_year (not used for now) """
        v_corpus = self.create_vectorized_corpus()
        Y, _ = v_corpus.collapse_by_category('year', aggregate_function='mean', dtype=np.float)
        expected_ytm = [np.array([4.0, 3.0, 7.0, 1.0]) / 2.0, np.array([6.0, 7.0, 4.0, 2.0]) / 3.0]
        self.assertTrue(np.allclose(expected_ytm, Y))

    def test_group_by_year_with_average(self):

        corpus = [
            "the house had a tiny little mouse",
            "the cat saw the mouse",
            "the mouse ran away from the house",
            "the cat finally ate the mouse",
            "the end of the mouse story",
        ]
        expected_bag_term_matrix = np.array(
            [
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 2, 0],
                [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0],
                [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0],
            ]
        )

        expected_bag_term_matrix_sums = np.array(
            [expected_bag_term_matrix[[0, 1, 2], :].sum(axis=0), expected_bag_term_matrix[[3, 4], :].sum(axis=0)]
        )

        expected_bag_term_matrix_means = np.array(
            [
                expected_bag_term_matrix[[0, 1, 2], :].sum(axis=0) / 3.0,
                expected_bag_term_matrix[[3, 4], :].sum(axis=0) / 2.0,
            ]
        )

        document_index = pd.DataFrame({'year': [1, 1, 1, 2, 2]})

        vectorizer = CountVectorizer()
        bag_term_matrix = vectorizer.fit_transform(corpus)

        v_corpus = vectorized_corpus.VectorizedCorpus(
            bag_term_matrix, token2id=vectorizer.vocabulary_, document_index=document_index
        )

        self.assertTrue(np.allclose(expected_bag_term_matrix, bag_term_matrix.todense()))

        y_sum_corpus = v_corpus.group_by_year2(aggregate_function='sum', dtype=np.float)
        y_mean_corpus = v_corpus.group_by_year2(aggregate_function='mean', dtype=np.float)

        self.assertTrue(np.allclose(expected_bag_term_matrix_sums, y_sum_corpus.data.todense()))
        self.assertTrue(np.allclose(expected_bag_term_matrix_means, y_mean_corpus.data.todense()))

        # token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3 }
        # bag_term_matrix = np.array([
        #     [2, 1, 4, 1],
        #     [2, 2, 3, 0],
        #     [2, 3, 2, 0],
        #     [2, 4, 1, 1],
        #     [2, 0, 1, 1]
        # ])
        # df = pd.DataFrame({'year': [ 2013, 2013, 2014, 2014, 2014 ]})
        # v_corpus = vectorized_corpus.VectorizedCorpus(bag_term_matrix, token2id, df)
        # return v_corpus

    def test_normalize_with_default_arguments_returns_matrix_normalized_by_l1_norm_for_each_row(self):
        bag_term_matrix = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
        token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        df = pd.DataFrame({'year': [2013, 2014]})
        v_corpus = vectorized_corpus.VectorizedCorpus(bag_term_matrix, token2id, df)
        n_corpus = v_corpus.normalize()
        E = np.array([[4, 3, 7, 1], [6, 7, 4, 2]]) / (np.array([[15, 19]]).T)
        self.assertTrue((E == n_corpus.bag_term_matrix).all())

    def test_normalize_with_keep_magnitude(self):
        bag_term_matrix = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
        bag_term_matrix = scipy.sparse.csr_matrix(bag_term_matrix)

        token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        df = pd.DataFrame({'year': [2013, 2014]})

        v_corpus = vectorized_corpus.VectorizedCorpus(bag_term_matrix, token2id, df)
        n_corpus = v_corpus.normalize(keep_magnitude=True)

        factor = 15.0 / 19.0
        E = np.array([[4.0, 3.0, 7.0, 1.0], [6.0 * factor, 7.0 * factor, 4.0 * factor, 2.0 * factor]])
        self.assertTrue(np.allclose(E, n_corpus.bag_term_matrix.todense()))

    def create_slice_by_n_count_test_corpus(self):
        bag_term_matrix = np.array([[1, 1, 4, 1], [0, 2, 3, 0], [0, 3, 2, 0], [0, 4, 1, 3], [2, 0, 1, 1]])
        token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        df = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
        return vectorized_corpus.VectorizedCorpus(bag_term_matrix, token2id, df)

    def test_slice_by_n_count_when_exists_tokens_below_count_returns_filtered_corpus(self):

        v_corpus = self.create_slice_by_n_count_test_corpus()

        # Act
        t_corpus = v_corpus.slice_by_n_count(6)

        # Assert
        expected_bag_term_matrix = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [0, 1]])

        self.assertEqual({'b': 0, 'c': 1}, t_corpus.token2id)
        self.assertEqual({'b': 10, 'c': 11}, t_corpus.word_counts)
        self.assertTrue((expected_bag_term_matrix == t_corpus.bag_term_matrix).all())

    def test_slice_by_n_count_when_all_below_below_n_count_returns_empty_corpus(self):

        v_corpus = self.create_slice_by_n_count_test_corpus()

        # Act
        t_corpus = v_corpus.slice_by_n_count(20)

        # Assert

        self.assertEqual({}, t_corpus.token2id)
        self.assertEqual({}, t_corpus.word_counts)
        self.assertTrue((np.empty((5, 0)) == t_corpus.bag_term_matrix).all())

    def test_slice_by_n_count_when_all_tokens_above_n_count_returns_same_corpus(self):

        v_corpus = self.create_slice_by_n_count_test_corpus()

        # Act
        t_corpus = v_corpus.slice_by_n_count(1)

        # Assert
        self.assertEqual(v_corpus.token2id, t_corpus.token2id)
        self.assertEqual(v_corpus.word_counts, t_corpus.word_counts)
        self.assertTrue(np.allclose(v_corpus.bag_term_matrix.todense().A, t_corpus.bag_term_matrix.todense().A))

    def test_slice_by_n_top_when_all_tokens_above_n_count_returns_same_corpus(self):

        v_corpus = self.create_slice_by_n_count_test_corpus()

        # Act
        t_corpus = v_corpus.slice_by_n_top(4)

        # Assert
        self.assertEqual(v_corpus.token2id, t_corpus.token2id)
        self.assertEqual(v_corpus.word_counts, t_corpus.word_counts)
        self.assertTrue(np.allclose(v_corpus.bag_term_matrix.todense().A, t_corpus.bag_term_matrix.todense().A))

    def test_slice_by_n_top_when_n_top_less_than_n_tokens_returns_corpus_with_top_n_counts(self):

        v_corpus = self.create_slice_by_n_count_test_corpus()

        # Act
        t_corpus = v_corpus.slice_by_n_top(2)

        # Assert
        expected_bag_term_matrix = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [0, 1]])

        self.assertEqual({'b': 0, 'c': 1}, t_corpus.token2id)
        self.assertEqual({'b': 10, 'c': 11}, t_corpus.word_counts)
        self.assertTrue((expected_bag_term_matrix == t_corpus.bag_term_matrix).all())

    def test_id2token_is_reversed_token2id(self):
        v_corpus = self.create_vectorized_corpus()
        id2token = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        self.assertEqual(id2token, v_corpus.id2token)
