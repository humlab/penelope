import uuid

import pytest  # pylint: disable=unused-import
from penelope.utility import (
    extract_filename_metadata,
    filename_satisfied_by,
    probe_extension,
    strip_extensions,
    strip_path_and_add_counter,
    strip_path_and_extension,
    strip_paths,
    touch,
)
from penelope.utility.filename_utils import replace_extension
from pyright import os

OUTPUT_FOLDER = './tests/output'


def test_touch():

    filename = f'./tests/output/{uuid.uuid1()}'
    assert os.path.isfile(touch(filename))

    filename = f'./tests/output/{uuid.uuid1()}/{uuid.uuid1()}'
    assert os.path.isfile(touch(filename))


def test_probe_extension():

    filename = f'./tests/output/{uuid.uuid1()}.txt'

    assert probe_extension(filename) is None
    assert probe_extension(filename, extensions='txt') is None

    touch(filename)
    assert probe_extension(filename) == filename
    assert probe_extension(filename, 'txt') == filename

    filename = replace_extension(filename, 'zip')
    assert probe_extension(filename, 'zip,csv,txt') == replace_extension(filename, 'txt')


def test_extract_filename_fields_when_valid_regexp_returns_metadata_values():
    filename = 'SOU 1957_5 Namn.txt'
    meta = extract_filename_metadata(
        filename=filename, filename_fields=dict(year=r".{4}(\d{4})_.*", serial_no=r".{8}_(\d+).*")
    )
    assert 5 == meta['serial_no']
    assert 1957 == meta['year']


def test_extract_filename_fields_when_indexed_split_returns_metadata_values():
    filename = 'prot_1957_5.txt'
    meta = extract_filename_metadata(filename=filename, filename_fields=["year:_:1", "serial_no:_:2"])
    assert 5 == meta['serial_no']
    assert 1957 == meta['year']


@pytest.mark.parametrize(
    'filename, unesco_id, year, city',
    [
        ('CONVENTION_0201_031038_2005_paris.txt', 31038, 2005, 'paris'),
        ('CONVENTION_0201_031038_2005.txt', 31038, 2005, None),
    ],
)
def test_extract_filename_fields_of_unesco_filename(filename, unesco_id, year, city):
    filename_fields = ["unesco_id:_:2", "year:_:3", r'city:\w+\_\d+\_\d+\_\d+\_(.*)\.txt']
    meta = extract_filename_metadata(filename=filename, filename_fields=filename_fields)
    assert unesco_id == meta['unesco_id']
    assert year == meta['year']
    assert city == meta['city']


def test_extract_filename_fields_when_invalid_regexp_returns_none():
    filename = 'xyz.txt'
    meta = extract_filename_metadata(filename=filename, filename_fields=dict(value=r".{4}(\d{4})_.*"))
    assert meta['value'] is None


def test_extract_filename_fields():
    # extract_filenames_metadata(filename, **kwargs)
    pass


def test_strip_path_and_extension():
    assert strip_path_and_extension('/tmp/hej.txt') == 'hej'
    assert strip_path_and_extension('/tmp/hej') == 'hej'
    assert strip_path_and_extension('hej') == 'hej'


def test_strip_path_and_add_counter():
    assert strip_path_and_add_counter('/tmp/hej.txt', 4, 3) == 'hej_004.txt'


def test_strip_extension():
    assert strip_extensions('/tmp/hej.txt') == '/tmp/hej'
    assert strip_extensions('/tmp/hej') == '/tmp/hej'
    assert strip_extensions('hej.x') == 'hej'


def test_strip_path():
    assert strip_paths('/tmp/hej.txt') == 'hej.txt'
    assert strip_paths(['/tmp/hej.txt']) == ['hej.txt']
    assert strip_paths('/tmp/hej') == 'hej'
    assert strip_paths('hej.x') == 'hej.x'


def test_filename_satisfied_by():

    assert filename_satisfied_by("abc.txt", filename_filter=None, filename_pattern=None)
    assert filename_satisfied_by("abc.txt", filename_filter=None, filename_pattern="*.txt")
    assert not filename_satisfied_by("abc.txt", filename_filter=None, filename_pattern="*.csv")
    assert filename_satisfied_by("abc.txt", filename_filter=None, filename_pattern="*.*")
    assert filename_satisfied_by("abc.txt", filename_filter=["abc.txt"], filename_pattern=None)
    assert not filename_satisfied_by("abc.txt", filename_filter=["abc.csv"], filename_pattern=None)
    assert filename_satisfied_by("abc.txt", filename_filter=lambda x: x in ["abc.txt"], filename_pattern=None)
    assert not filename_satisfied_by("abc.txt", filename_filter=lambda x: x not in ["abc.txt"], filename_pattern=None)

    assert filename_satisfied_by("abc", filename_filter=["abc"], filename_pattern=None)
    # assert filename_satisfied_by("abc", filename_filter=["abc.txt"], filename_pattern=None)


def test_basename():
    # basename(path)
    pass


def test_store():
    # utility.store(archive_name: str, stream: Iterable[Tuple[str,Iterable[str]]])
    pass


def test_read():
    # utility.read(folder_or_zip: Union[str, zipfile.ZipFile], filename: str, as_binary=False)
    pass


def test_read_textfile():
    # utility.read_textfile(filename)
    pass


def test_filename_field_parser():
    # utility.filename_field_parser(meta_fields)
    pass
