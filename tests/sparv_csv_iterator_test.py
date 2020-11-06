import os

import penelope.corpus.readers as readers
from penelope.corpus.readers import AnnotationOpts

SPARV_CSV_EXPORT_FILENAME = './tests/test_data/prot_197677__27.tsv'
SPARV_CSV_EXPORT_FILENAME_SMALL = './tests/test_data/sparv_csv_export_small.csv'
SPARV_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/sparv_zipped_csv_export.zip'


def sparv_csv_export_text():
    with open(SPARV_CSV_EXPORT_FILENAME, "r") as fp:
        return fp.read()


def sparv_csv_export_small_text():
    with open(SPARV_CSV_EXPORT_FILENAME_SMALL, "r") as fp:
        return fp.read()


def test_reader_when_no_transforms_returns_source_tokens():

    tokenizer = readers.SparvCsvTokenizer(
        source=SPARV_CSV_EXPORT_FILENAME_SMALL,
        annotation_opts=AnnotationOpts(pos_includes=None, pos_excludes=None, lemmatize=False),
    )

    expected = "Rödräven är ett hunddjur som har en mycket vidsträckt utbredning över norra halvklotet .".split()

    filename, tokens = next(tokenizer)

    assert filename == os.path.split(filename)[1]
    assert expected == tokens


def test_reader_when_only_nn_returns_only_nn():

    tokenizer = readers.SparvCsvTokenizer(
        source=SPARV_CSV_EXPORT_FILENAME_SMALL,
        annotation_opts=AnnotationOpts(pos_includes='NN', pos_excludes=None, lemmatize=False),
    )

    expected = "Rödräven hunddjur utbredning halvklotet".split()

    filename, tokens = next(tokenizer)

    assert filename == os.path.split(filename)[1]
    assert expected == tokens


def test_reader_when_lemmatized_nn_returns_lemmatized_nn():

    tokenizer = readers.SparvCsvTokenizer(
        source=SPARV_CSV_EXPORT_FILENAME_SMALL,
        annotation_opts=AnnotationOpts(pos_includes='NN', pos_excludes=None, lemmatize=True),
    )

    expected = "rödräv hunddjur utbredning halvklot".split()

    filename, tokens = next(tokenizer)

    assert filename == os.path.split(filename)[1]
    assert expected == tokens


def test_reader_when_lemmatized_nn_vb_returns_lemmatized_nn_vb():

    tokenizer = readers.SparvCsvTokenizer(
        source=SPARV_CSV_EXPORT_FILENAME_SMALL,
        annotation_opts=AnnotationOpts(pos_includes='NN|VB', pos_excludes=None, lemmatize=True),
    )

    expected = "rödräv vara hunddjur ha utbredning halvklot".split()

    filename, tokens = next(tokenizer)

    assert filename == os.path.split(filename)[1]
    assert expected == tokens


def test_reader_when_lemmatized_nnvb_pos_appended_returns_lemmatized_nn_vb_pos():

    tokenizer = readers.SparvCsvTokenizer(
        source=SPARV_CSV_EXPORT_FILENAME_SMALL,
        annotation_opts=AnnotationOpts(pos_includes='NN|VB', pos_excludes=None, lemmatize=True, append_pos=True),
    )

    expected = "rödräv|NN vara|VB hunddjur|NN ha|VB utbredning|NN halvklot|NN".split()

    filename, tokens = next(tokenizer)

    assert filename == os.path.split(filename)[1]
    assert expected == tokens


def test_reader_when_source_is_zipped_archive_succeeds():

    expected_documents = [
        ['rödräv', 'hunddjur', 'utbredning', 'halvklot'],
    ]
    expected_names = ["sparv_1978_001.txt"]

    reader = readers.SparvCsvTokenizer(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME,
        chunk_size=None,
        annotation_opts=AnnotationOpts(
            pos_includes='|NN|',
            lemmatize=True,
        ),
    )

    for i, (document_name, tokens) in enumerate(reader):

        assert expected_documents[i] == list(tokens)
        assert expected_names[i] == document_name
