import unittest

import penelope.corpus.tokenized_corpus as corpora
from tests.utils import create_text_tokenizer


class Test_ProcessedCorpus(unittest.TestCase):
    def setUp(self):
        pass

    def create_reader(self):
        filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_tokenizer(filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True)
        return reader

    def test_next_document_when_only_any_alphanumeric_is_false_returns_all_tokens(self):
        reader = self.create_reader()
        kwargs = dict(
            only_any_alphanumeric=False,
            to_lower=False,
            remove_accents=False,
            min_len=1,
            max_len=None,
            keep_numerals=True,
            only_alphabetic=False,
        )
        corpus = corpora.TokenizedCorpus(reader, **kwargs)
        _, tokens = next(corpus)
        expected = [
            "Tre",
            "svarta",
            "ekar",
            "ur",
            "snön",
            ".",
            "Så",
            "grova",
            ",",
            "men",
            "fingerfärdiga",
            ".",
            "Ur",
            "deras",
            "väldiga",
            "flaskor",
            "ska",
            "grönskan",
            "skumma",
            "i",
            "vår",
            ".",
        ]
        self.assertEqual(expected, tokens)

    def test_next_document_when_only_any_alphanumeric_true_skips_deliminators(self):
        reader = self.create_reader()
        corpus = corpora.TokenizedCorpus(
            reader, only_any_alphanumeric=True, to_lower=False, remove_accents=False, min_len=1, keep_numerals=True
        )
        _, tokens = next(corpus)
        expected = [
            "Tre",
            "svarta",
            "ekar",
            "ur",
            "snön",
            "Så",
            "grova",
            "men",
            "fingerfärdiga",
            "Ur",
            "deras",
            "väldiga",
            "flaskor",
            "ska",
            "grönskan",
            "skumma",
            "i",
            "vår",
        ]
        self.assertEqual(expected, tokens)

    def test_next_document_when_only_any_alphanumeric_true_skips_deliminators_using_defaults(self):
        reader = create_text_tokenizer(filename_fields=None, fix_whitespaces=True, fix_hyphenation=True)
        corpus = corpora.TokenizedCorpus(reader, only_any_alphanumeric=True)
        _, tokens = next(corpus)
        expected = [
            "Tre",
            "svarta",
            "ekar",
            "ur",
            "snön",
            "Så",
            "grova",
            "men",
            "fingerfärdiga",
            "Ur",
            "deras",
            "väldiga",
            "flaskor",
            "ska",
            "grönskan",
            "skumma",
            "i",
            "vår",
        ]
        self.assertEqual(expected, tokens)

    def test_next_document_when_to_lower_is_true_returns_all_lowercase(self):
        reader = self.create_reader()
        kwargs = dict(
            only_any_alphanumeric=True, to_lower=True, remove_accents=False, min_len=1, max_len=None, keep_numerals=True
        )
        corpus = corpora.TokenizedCorpus(reader, **kwargs)
        _, tokens = next(corpus)
        expected = [
            "tre",
            "svarta",
            "ekar",
            "ur",
            "snön",
            "så",
            "grova",
            "men",
            "fingerfärdiga",
            "ur",
            "deras",
            "väldiga",
            "flaskor",
            "ska",
            "grönskan",
            "skumma",
            "i",
            "vår",
        ]
        self.assertEqual(expected, tokens)

    def test_next_document_when_min_len_is_two_returns_single_char_words_filtered_out(self):
        reader = self.create_reader()
        kwargs = dict(
            only_any_alphanumeric=True, to_lower=True, remove_accents=False, min_len=2, max_len=None, keep_numerals=True
        )
        corpus = corpora.TokenizedCorpus(reader, **kwargs)
        _, tokens = next(corpus)
        expected = [
            "tre",
            "svarta",
            "ekar",
            "ur",
            "snön",
            "så",
            "grova",
            "men",
            "fingerfärdiga",
            "ur",
            "deras",
            "väldiga",
            "flaskor",
            "ska",
            "grönskan",
            "skumma",
            "vår",
        ]
        self.assertEqual(expected, tokens)

    def test_next_document_when_max_len_is_six_returns_filter_out_longer_words(self):
        reader = self.create_reader()
        kwargs = dict(
            only_any_alphanumeric=True, to_lower=True, remove_accents=False, min_len=2, max_len=6, keep_numerals=True
        )
        corpus = corpora.TokenizedCorpus(reader, **kwargs)
        _, tokens = next(corpus)
        expected = ["tre", "svarta", "ekar", "ur", "snön", "så", "grova", "men", "ur", "deras", "ska", "skumma", "vår"]
        self.assertEqual(expected, tokens)

    def test_n_tokens_when_exhausted_and_only_any_alphanumeric_min_len_two_returns_expected_count(self):
        reader = self.create_reader()
        corpus = corpora.TokenizedCorpus(reader, only_any_alphanumeric=True, min_len=2)
        n_expected = [17, 13, 21, 42, 18]
        _ = [x for x in corpus]
        n_tokens = list(corpus.documents.n_tokens)
        self.assertEqual(n_expected, n_tokens)

    def test_next_document_when_new_corpus_returns_document(self):
        reader = create_text_tokenizer(fix_whitespaces=True, fix_hyphenation=True)
        corpus = corpora.TokenizedCorpus(reader)
        result = next(corpus)
        expected = (
            "Tre svarta ekar ur snön . "
            + "Så grova , men fingerfärdiga . "
            + "Ur deras väldiga flaskor "
            + "ska grönskan skumma i vår ."
        )
        self.assertEqual(expected, ' '.join(result[1]))

    def test_get_index_when_extract_passed_returns_metadata(self):
        filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_tokenizer(filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True)
        corpus = corpora.TokenizedCorpus(reader)
        result = corpus.metadata
        expected = [
            dict(filename='dikt_2019_01_test.txt', serial_no=1, year=2019),
            dict(filename='dikt_2019_02_test.txt', serial_no=2, year=2019),
            dict(filename='dikt_2019_03_test.txt', serial_no=3, year=2019),
            dict(filename='dikt_2020_01_test.txt', serial_no=1, year=2020),
            dict(filename='dikt_2020_02_test.txt', serial_no=2, year=2020),
        ]
        self.assertEqual(len(expected), len(result))
        for i in range(0, len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_get_index_when_no_extract_passed_returns_not_none(self):
        reader = create_text_tokenizer(filename_fields=None, fix_whitespaces=True, fix_hyphenation=True)
        corpus = corpora.TokenizedCorpus(reader)
        result = corpus.metadata
        self.assertIsNotNone(result)

    def test_next_document_when_token_corpus_returns_tokenized_document(self):
        reader = create_text_tokenizer(filename_fields=None, fix_whitespaces=True, fix_hyphenation=True)
        corpus = corpora.TokenizedCorpus(reader, only_any_alphanumeric=False)
        _, tokens = next(corpus)
        expected = [
            "Tre",
            "svarta",
            "ekar",
            "ur",
            "snön",
            ".",
            "Så",
            "grova",
            ",",
            "men",
            "fingerfärdiga",
            ".",
            "Ur",
            "deras",
            "väldiga",
            "flaskor",
            "ska",
            "grönskan",
            "skumma",
            "i",
            "vår",
            ".",
        ]
        self.assertEqual(expected, tokens)

    def test_get_index_when_extract_passed_returns_expected_count(self):
        reader = self.create_reader()
        kwargs = dict(
            only_any_alphanumeric=False,
            to_lower=False,
            remove_accents=False,
            min_len=2,
            max_len=None,
            keep_numerals=True,
        )
        corpus = corpora.TokenizedCorpus(reader, **kwargs)
        result = corpus.metadata
        self.assertEqual(5, len(result))

    def test_n_tokens_when_exhausted_iterater_returns_expected_count(self):
        reader = self.create_reader()
        corpus = corpora.TokenizedCorpus(reader, only_any_alphanumeric=False)
        _ = [x for x in corpus]
        n_tokens = list(corpus.documents.n_tokens)
        expected = [22, 16, 26, 45, 21]
        self.assertEqual(expected, n_tokens)

    def test_n_tokens_when_exhausted_and_only_any_alphanumeric_is_true_returns_expected_count(self):
        reader = create_text_tokenizer(filename_fields=None, fix_whitespaces=True, fix_hyphenation=True)
        corpus = corpora.TokenizedCorpus(reader, only_any_alphanumeric=True)
        _ = [x for x in corpus]
        n_tokens = list(corpus.documents.n_tokens)
        expected = [18, 14, 24, 42, 18]
        self.assertEqual(expected, n_tokens)


    def test_corpus_can_be_reiterated(self):

        reader = create_text_tokenizer(filename_fields=None, fix_whitespaces=True, fix_hyphenation=True)

        corpus = corpora.TokenizedCorpus(reader, only_any_alphanumeric=True)
        for i in range(0,4):
            n_tokens = [len(x) for x in corpus.terms]
            expected = [18, 14, 24, 42, 18]
            self.assertEqual(expected, n_tokens, f"iteration{i}")

