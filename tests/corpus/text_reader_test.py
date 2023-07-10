from penelope import corpus as pc
from penelope.corpus.readers import TextReader

from .. import utils as tu


def simple_test_corpus(
    filename_fields=None,
    filename_filter: str = None,
    filename_pattern: str = "*.txt",
    text_transforms: str = "dehyphen,normalize-whitespace",
) -> TextReader:
    reader: TextReader = TextReader(
        source=tu.TEST_CORPUS_FILENAME,
        reader_opts=pc.TextReaderOpts(
            filename_pattern=filename_pattern,
            filename_filter=filename_filter,
            filename_fields=filename_fields,
        ),
        transform_opts=pc.TextTransformOpts(transforms=text_transforms),
    )
    return reader


def get_file(reader, filename):
    for name, tokens in reader:
        if name == filename:
            return tokens
    raise FileNotFoundError(filename)


def test_archive_filenames_when_filter_txt_returns_txt_files():
    reader = simple_test_corpus(filename_pattern='*.txt')
    assert 5 == len(reader.filenames)


def test_archive_filenames_when_filter_md_returns_md_files():
    reader = simple_test_corpus(filename_pattern='*.md')
    assert 1 == len(reader.filenames)


def test_archive_filenames_when_filter_function_txt_returns_txt_files():
    def filename_filter(x):
        return x.endswith('txt')

    reader = simple_test_corpus(filename_filter=filename_filter)
    assert 5 == len(reader.filenames)


def test_get_file_when_default_returns_unmodified_content():
    filename = 'dikt_2019_01_test.txt'
    reader = simple_test_corpus(text_transforms="dehyphen,normalize-whitespace", filename_filter=[filename])
    result = next(reader)
    expected = (
        "Tre svarta ekar ur snön.\nSå grova, men fingerfärdiga.\nUr deras väldiga flaskor\nska grönskan skumma i vår."
    )
    assert filename == result[0]
    assert expected == result[1]


def test_metadata_has_filename():
    reader = simple_test_corpus()
    assert reader is not None
    assert reader.filenames is not None
    assert len(reader.filenames) > 0
    assert len(reader.metadata) > 0
    assert len(reader.metadata[0].keys()) > 0
    assert 'filename' in reader.metadata[0]


def test_can_get_file_when_compress_whitespace_is_true_strips_whitespaces():
    filename = 'dikt_2019_01_test.txt'
    reader = simple_test_corpus(text_transforms="normalize-whitespace", filename_filter=[filename])
    result = next(reader)
    expected = (
        "Tre svarta ekar ur snön.\nSå grova, men fingerfärdiga.\nUr deras väldiga flaskor\nska grönskan skumma i vår."
    )
    assert filename == result[0]
    assert expected == result[1]


def test_get_file_when_fix_hyphenation_is_true_removes_hyphens():
    filename = 'dikt_2019_03_test.txt'
    reader = simple_test_corpus(text_transforms="dehyphen,normalize-whitespace", filename_filter=[filename])
    result = next(reader)
    expected = 'Nordlig storm. Det är den i den tid när rönnbärsklasar\nmognar. Vaken i mörkret hör man\nstjärnbilderna stampa i sina spiltor\nhögt över trädet'
    assert filename == result[0]
    assert expected == result[1]


def test_get_file_when_file_exists_and_extractor_specified_returns_content_and_metadata():
    filename = 'dikt_2019_03_test.txt'
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = simple_test_corpus(
        filename_fields=filename_fields, text_transforms="dehyphen,normalize-whitespace", filename_filter=[filename]
    )
    result = next(reader)
    expected = 'Nordlig storm. Det är den i den tid när rönnbärsklasar\nmognar. Vaken i mörkret hör man\nstjärnbilderna stampa i sina spiltor\nhögt över trädet'
    assert filename == result[0]
    assert expected == result[1]
    assert reader.metadata[0]['year'] > 0


def test_get_index_when_filename_fields_is_set_returns_metadata():
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = simple_test_corpus(filename_fields=filename_fields, text_transforms="dehyphen,normalize-whitespace")
    result = reader.metadata
    expected = [
        dict(filename='dikt_2019_01_test.txt', serial_no=1, year=2019),
        dict(filename='dikt_2019_02_test.txt', serial_no=2, year=2019),
        dict(filename='dikt_2019_03_test.txt', serial_no=3, year=2019),
        dict(filename='dikt_2020_01_test.txt', serial_no=1, year=2020),
        dict(filename='dikt_2020_02_test.txt', serial_no=2, year=2020),
    ]

    assert len(expected) == len(result)
    for i in range(0, len(expected)):
        assert expected[i] == result[i]


def test_get_index_when_extractor_passed_returns_metadata2():
    filename_fields = "year:_:1#serial_no:_:2"
    reader: TextReader = simple_test_corpus(
        filename_fields=filename_fields, text_transforms="dehyphen,normalize-whitespace"
    )
    result = reader.metadata
    expected = [
        dict(filename='dikt_2019_01_test.txt', serial_no=1, year=2019),
        dict(filename='dikt_2019_02_test.txt', serial_no=2, year=2019),
        dict(filename='dikt_2019_03_test.txt', serial_no=3, year=2019),
        dict(filename='dikt_2020_01_test.txt', serial_no=1, year=2020),
        dict(filename='dikt_2020_02_test.txt', serial_no=2, year=2020),
    ]

    assert len(expected) == len(result)
    for i in range(0, len(expected)):
        assert expected[i] == result[i]

    reader.apply_filter(['dikt_2019_01_test.txt', 'dikt_2019_02_test.txt'])

    result = reader.metadata
    expected = [
        dict(filename='dikt_2019_01_test.txt', serial_no=1, year=2019),
        dict(filename='dikt_2019_02_test.txt', serial_no=2, year=2019),
    ]

    assert len(expected) == len(result)
    for i in range(0, len(expected)):
        assert expected[i] == result[i]


def test_reader_can_be_reiterated():
    reader: TextReader = simple_test_corpus(filename_fields="year:_:1", text_transforms="dehyphen,normalize-whitespace")
    for _ in range(0, 4):
        n_chars = [len(x) for _, x in reader]
        expected = [105, 84, 140, 220, 93]
        assert expected == n_chars
