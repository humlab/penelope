import pytest  # pylint: disable=unused-import
from penelope.corpus import readers
from penelope.corpus.readers.text_tokenizer import TextTokenizer
from tests.utils import create_tokens_reader


def get_file(reader, filename):
    for name, tokens in reader:
        if name == filename:
            return tokens
    raise FileNotFoundError(filename)


def test_archive_filenames_when_filter_txt_returns_txt_files():
    reader = create_tokens_reader(filename_pattern='*.txt')
    assert 5 == len(reader.filenames)


def test_archive_filenames_when_filter_md_returns_md_files():
    reader = create_tokens_reader(filename_pattern='*.md')
    assert 1 == len(reader.filenames)


def test_archive_filenames_when_filter_function_txt_returns_txt_files():
    def filename_filter(x):
        return x.endswith('txt')

    reader = create_tokens_reader(filename_filter=filename_filter)
    assert 5 == len(reader.filenames)


def test_tokenize_corpus_with_list_source():

    source = readers.TextTokenizer(source=["a b c", "e f g"])

    assert [('document_1.txt', ['a', 'b', 'c']), ('document_2.txt', ['e', 'f', 'g'])] == [x for x in source]


def test_get_file_when_default_returns_unmodified_content():
    filename = 'dikt_2019_01_test.txt'
    reader = create_tokens_reader(fix_whitespaces=False, fix_hyphenation=True, filename_filter=[filename])
    result = next(reader)
    expected = (
        "Tre svarta ekar ur snön . "
        + "Så grova , men fingerfärdiga . "
        + "Ur deras väldiga flaskor "
        + "ska grönskan skumma i vår ."
    )
    assert filename == result[0]
    assert expected == ' '.join(result[1])


def test_metadata_has_filena():
    tokens_reader = create_tokens_reader()
    assert tokens_reader is not None
    assert tokens_reader.filenames is not None
    assert len(tokens_reader.filenames) > 0
    assert len(tokens_reader.metadata) > 0
    assert len(tokens_reader.metadata[0].keys()) > 0
    assert 'filename' in tokens_reader.metadata[0]


def test_can_get_file_when_compress_whitespace_is_true_strips_whitespaces():
    filename = 'dikt_2019_01_test.txt'
    reader = create_tokens_reader(fix_whitespaces=True, fix_hyphenation=True, filename_filter=[filename])
    result = next(reader)
    expected = (
        "Tre svarta ekar ur snön . "
        + "Så grova , men fingerfärdiga . "
        + "Ur deras väldiga flaskor "
        + "ska grönskan skumma i vår ."
    )
    assert filename == result[0]
    assert expected == ' '.join(result[1])


def test_get_file_when_fix_hyphenation_is_trye_removes_hyphens():
    filename = 'dikt_2019_03_test.txt'
    reader = create_tokens_reader(fix_whitespaces=True, fix_hyphenation=True, filename_filter=[filename])
    result = next(reader)
    expected = (
        "Nordlig storm . Det är den i den tid när rönnbärsklasar mognar . Vaken i mörkret hör man "
        + "stjärnbilderna stampa i sina spiltor "
        + "högt över trädet"
    )
    assert filename == result[0]
    assert expected == ' '.join(result[1])


def test_get_file_when_file_exists_and_extractor_specified_returns_content_and_metadata():
    filename = 'dikt_2019_03_test.txt'
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = create_tokens_reader(
        filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True, filename_filter=[filename]
    )
    result = next(reader)
    expected = (
        "Nordlig storm . Det är den i den tid när rönnbärsklasar mognar . Vaken i mörkret hör man "
        + "stjärnbilderna stampa i sina spiltor "
        + "högt över trädet"
    )
    assert filename == result[0]
    assert expected == ' '.join(result[1])
    assert reader.metadata[0]['year'] > 0


def test_get_index_when_extractor_passed_returns_metadata():
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = create_tokens_reader(filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True)
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
    reader: TextTokenizer = create_tokens_reader(
        filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True
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

    reader: TextTokenizer = create_tokens_reader(filename_fields="year:_:1", fix_whitespaces=True, fix_hyphenation=True)
    for _ in range(0, 4):
        n_tokens = [len(x) for _, x in reader]
        expected = [22, 16, 26, 45, 21]
        assert expected == n_tokens
