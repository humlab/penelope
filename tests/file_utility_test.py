import pytest  # pylint: disable=unused-import

import penelope.utility.file_utility as file_utility

from tests.utils import TEST_CORPUS_FILENAME


def test_extract_filename_fields_when_valid_regexp_returns_metadata_values():
    filename = 'SOU 1957_5 Namn.txt'
    meta = file_utility.extract_filename_fields(filename, year=r".{4}(\d{4})_.*", serial_no=r".{8}_(\d+).*")
    assert 5 == meta.serial_no
    assert 1957 == meta.year


def test_extract_filename_fields_when_invalid_regexp_returns_none():
    filename = 'xyz.txt'
    meta = file_utility.extract_filename_fields(filename, value=r".{4}(\d{4})_.*")
    assert meta.value is None


def test_extract_filename_fields():
    # file_utility.extract_filename_fields(filename, **kwargs)
    pass


def test_strip_path_and_extension():
    # file_utility.strip_path_and_extension(filename)
    pass


def test_strip_path_and_add_counter():
    # file_utility.strip_path_and_add_counter(filename, n_chunk)
    pass


def test_filename_satisfied_by():
    # file_utility.filename_satisfied_by(filename, filename_filter)
    pass


def test_basename():
    # basename(path)
    pass


def test_create_iterator():
    stream = file_utility.create_iterator(
        TEST_CORPUS_FILENAME, ['dikt_2019_01_test.txt'], filename_pattern='*.txt', as_binary=False
    )
    assert len([x for x in stream]) == 1


def test_list_filenames_when_source_is_a_filename():

    filenames = file_utility.list_filenames(TEST_CORPUS_FILENAME, filename_pattern='*.txt', filename_filter=None)
    assert len(filenames) == 5

    filenames = file_utility.list_filenames(TEST_CORPUS_FILENAME, filename_pattern='*.dat', filename_filter=None)
    assert len(filenames) == 0

    filenames = file_utility.list_filenames(
        TEST_CORPUS_FILENAME, filename_pattern='*.txt', filename_filter=['dikt_2019_01_test.txt']
    )
    assert len(filenames) == 1

    filenames = file_utility.list_filenames(
        TEST_CORPUS_FILENAME, filename_pattern='*.txt', filename_filter=lambda x: x == 'dikt_2019_01_test.txt'
    )
    assert len(filenames) == 1


def list_filenames_when_source_is_folder():
    # file_utility.list_filenames(folder_or_zip, filename_pattern: str="*.txt", filename_filter: Union[List[str],Callable]=None)
    pass


def test_store():
    # file_utility.store(archive_name: str, stream: Iterable[Tuple[str,Iterable[str]]])
    pass


def test_read():
    # file_utility.read(folder_or_zip: Union[str, zipfile.ZipFile], filename: str, as_binary=False)
    pass


def test_read_textfile():
    # file_utility.read_textfile(filename)
    pass


def test_filename_field_parser():
    # file_utility.filename_field_parser(meta_fields)
    pass
