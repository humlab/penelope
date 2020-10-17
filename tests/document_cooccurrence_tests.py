import pytest  # pylint: disable=unused-import

import penelope  # pylint: disable=unused-import
from penelope.corpus.vectorizer import CorpusVectorizer

# from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus
# from penelope.corpus.windowed_corpus import (concept_windows, corpus_concept_windows)


SPARV_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/tranströmer_corpus_export.csv.zip'


def test_cooccurrence():

    windows = [
        ["tran_2019_01_test.txt", 0, ["*", "*", "kyrka", "tränga", "turist"]],
        ["tran_2019_01_test.txt", 1, ["turist", "halvmörker", "valv", "gapa", "valv"]],
        ["tran_2019_01_test.txt", 2, ["valv", "gapa", "valv", "överblick", "ljuslåga"]],
        ["tran_2019_01_test.txt", 3, ["människa", "öppna", "valv", "valv", "bli"]],
        ["tran_2019_01_test.txt", 4, ["öppna", "valv", "valv", "bli", "vara"]],
        ["tran_2019_01_test.txt", 5, ["skola", "tår", "piazza", "Mr", "Mrs"]],
        ["tran_2019_01_test.txt", 6, ["signora", "öppna", "valv", "valv", "*"]],
        ["tran_2019_01_test.txt", 7, ["öppna", "valv", "valv", "*", "*"]],
    ]
    expected_windows = None

    _ = CorpusVectorizer()
    assert expected_windows == windows
    # # log_corpus_windows_fixture('./tests/test_data/corpus_windows_2_nnvb_lemma.json', corpus, windows)
    # corpus = SparvTokenizedCsvCorpus(SPARV_ZIPPED_CSV_EXPORT_FILENAME, pos_includes='|NN|VB|', lemmatize=True)
