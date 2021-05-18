import zipfile

from penelope.utility import zip_utils


def test_read_file_content():
    zip_filename = './tests/test_data/test_corpus.zip'
    filename = "README.md"
    data = zip_utils.read_file_content(zip_or_filename=zip_filename, filename=filename)
    assert data == "apa"
    with zipfile.ZipFile(zip_filename) as zf:
        data = zip_utils.read_file_content(zip_or_filename=zf, filename=filename)
    assert data == "apa"


def test_read_file_content2():
    zip_filename = './tests/test_data/test_corpus.zip'
    filename = "README.md"
    data = zip_utils.read_file_content2(zip_or_filename=zip_filename, filename=filename)
    assert data == (filename, "apa")


def test_list_filenames():
    expected_text_files = [
        'dikt_2019_01_test.txt',
        'dikt_2019_02_test.txt',
        'dikt_2019_03_test.txt',
        'dikt_2020_01_test.txt',
        'dikt_2020_02_test.txt',
    ]

    zip_filename = './tests/test_data/test_corpus.zip'
    filenames = zip_utils.list_filenames(zip_or_filename=zip_filename)
    assert filenames == expected_text_files


def test_store():
    ...


def test_compress():
    ...


def unpack():
    ...


def test_read_json():
    ...


def read_dataframe():
    ...
