import pytest  # pylint: disable=unused-import

from penelope.corpus.sparv.sparv_csv_to_text import SparvCsvToText

SPARV_CSV_EXPORT_FILENAME_SMALL = './tests/test_data/sparv_csv_export_small.csv'


def sparv_csv_export_small_text():
    with open(SPARV_CSV_EXPORT_FILENAME_SMALL, "r") as fp:
        return fp.read()


TEST_DATA = sparv_csv_export_small_text()


def test_reader_when_no_transforms_returns_source_tokens():

    reader = SparvCsvToText(pos_includes=None, pos_excludes=None, lemmatize=False, append_pos=False)

    expected = "Rödräven är ett hunddjur som har en mycket vidsträckt utbredning över norra halvklotet ."

    result = reader.transform(TEST_DATA)

    assert expected == result


def test_reader_when_only_nn_returns_only_nn():

    reader = SparvCsvToText(pos_includes='NN', pos_excludes=None, lemmatize=False, append_pos=False)

    expected = "Rödräven hunddjur utbredning halvklotet"

    result = reader.transform(TEST_DATA)

    assert expected == result


def test_reader_when_lemmatized_nn_returns_lemmatized_nn():

    reader = SparvCsvToText(pos_includes='NN', pos_excludes=None, lemmatize=True, append_pos=False)

    expected = "rödräv hunddjur utbredning halvklot"

    result = reader.transform(TEST_DATA)

    assert expected == result


def test_reader_when_lemmatized_nn_vb_returns_lemmatized_nn_vb():

    reader = SparvCsvToText(pos_includes='NN|VB', pos_excludes=None, lemmatize=True, append_pos=False)

    expected = "rödräv vara hunddjur ha utbredning halvklot"

    result = reader.transform(TEST_DATA)

    assert expected == result


def test_reader_when_lemmatized_nn_vb_pos_appendedreturns_lemmatized_nn_vb_pos():

    reader = SparvCsvToText(pos_includes='NN|VB', pos_excludes=None, lemmatize=True, append_pos=True)

    expected = "rödräv|NN vara|VB hunddjur|NN ha|VB utbredning|NN halvklot|NN"

    result = reader.transform(TEST_DATA)

    assert expected == result
