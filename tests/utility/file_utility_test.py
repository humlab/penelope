import uuid

import pytest  # pylint: disable=unused-import
from pyright import os

from penelope import utility as pu
from penelope.utility.filename_utils import replace_extension

OUTPUT_FOLDER = './tests/output'


def test_touch():
    filename = f'./tests/output/{uuid.uuid1()}'
    assert os.path.isfile(pu.touch(filename))

    filename = f'./tests/output/{uuid.uuid1()}/{uuid.uuid1()}'
    assert os.path.isfile(pu.touch(filename))


def test_probe_extension():
    filename = f'./tests/output/{uuid.uuid1()}.txt'

    assert pu.probe_extension(filename) is None
    assert pu.probe_extension(filename, extensions='txt') is None

    pu.touch(filename)
    assert pu.probe_extension(filename) == filename
    assert pu.probe_extension(filename, 'txt') == filename

    filename = replace_extension(filename, 'zip')
    assert pu.probe_extension(filename, 'zip,csv,txt') == replace_extension(filename, 'txt')


def test_extract_filename_fields_when_valid_regexp_returns_metadata_values():
    filename = 'SOU 1957_5 Namn.txt'
    meta = pu.extract_filename_metadata(
        filename=filename, filename_fields=dict(year=r".{4}(\d{4})_.*", serial_no=r".{8}_(\d+).*")
    )
    assert 5 == meta['serial_no']
    assert 1957 == meta['year']


def test_extract_filename_fields_when_indexed_split_returns_metadata_values():
    filename = 'prot_1957_5.txt'
    meta = pu.extract_filename_metadata(filename=filename, filename_fields=["year:_:1", "serial_no:_:2"])
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
    meta = pu.extract_filename_metadata(filename=filename, filename_fields=filename_fields)
    assert unesco_id == meta['unesco_id']
    assert year == meta['year']
    assert city == meta['city']


def test_extract_filename_fields_when_invalid_regexp_returns_none():
    filename = 'xyz.txt'
    meta = pu.extract_filename_metadata(filename=filename, filename_fields=dict(value=r".{4}(\d{4})_.*"))
    assert meta['value'] is None


def test_extract_filename_fields():
    # extract_filenames_metadata(filename, **kwargs)
    pass


def test_strip_path_and_extension():
    assert pu.strip_path_and_extension('/tmp/hej.txt') == 'hej'
    assert pu.strip_path_and_extension('/tmp/hej') == 'hej'
    assert pu.strip_path_and_extension('hej') == 'hej'
    assert pu.strip_path_and_extension(['/tmp/hej.txt', 'då']) == ['hej', 'då']
    assert pu.strip_path_and_extension([]) == []


def test_strip_path_and_add_counter():
    assert pu.strip_path_and_add_counter('/tmp/hej.txt', 4, 3) == 'hej_004.txt'


def test_strip_extension():
    assert pu.strip_extensions('/tmp/hej.txt') == '/tmp/hej'
    assert pu.strip_extensions('/tmp/hej') == '/tmp/hej'
    assert pu.strip_extensions('hej.x') == 'hej'


def test_strip_path():
    assert pu.strip_paths('/tmp/hej.txt') == 'hej.txt'
    assert pu.strip_paths(['/tmp/hej.txt']) == ['hej.txt']
    assert pu.strip_paths('/tmp/hej') == 'hej'
    assert pu.strip_paths('hej.x') == 'hej.x'


def test_filename_satisfied_by():
    assert pu.filename_satisfied_by("abc.txt", filename_filter=None, filename_pattern=None)
    assert pu.filename_satisfied_by("abc.txt", filename_filter=None, filename_pattern="*.txt")
    assert not pu.filename_satisfied_by("abc.txt", filename_filter=None, filename_pattern="*.csv")
    assert pu.filename_satisfied_by("abc.txt", filename_filter=None, filename_pattern="*.*")
    assert pu.filename_satisfied_by("abc.txt", filename_filter=["abc.txt"], filename_pattern=None)
    assert not pu.filename_satisfied_by("abc.txt", filename_filter=["abc.csv"], filename_pattern=None)
    assert pu.filename_satisfied_by("abc.txt", filename_filter=lambda x: x in ["abc.txt"], filename_pattern=None)
    assert not pu.filename_satisfied_by(
        "abc.txt", filename_filter=lambda x: x not in ["abc.txt"], filename_pattern=None
    )

    assert pu.filename_satisfied_by("abc", filename_filter=["abc"], filename_pattern=None)
    # assert filename_satisfied_by("abc", filename_filter=["abc.txt"], filename_pattern=None)


def test_replace_folder():
    assert pu.replace_folder("abc.txt", "tmp") == "tmp/abc.txt"
    assert pu.replace_folder(["abc.txt"], "tmp") == ["tmp/abc.txt"]
    assert pu.replace_folder("abc.txt", None) == "abc.txt"
    assert pu.replace_folder(["abc.txt"], None) == ["abc.txt"]
    assert pu.replace_folder("tmp/abc.txt", None) == "abc.txt"

def test_replace_extension():
    assert pu.replace_extension("abc.txt", "csv") == "abc.csv"  
    assert pu.replace_extension(["abc.txt"], "csv") == ["abc.csv"]
    assert pu.replace_extension("abc.txt", None) == "abc"
    assert pu.replace_extension(["abc.txt"], None) == ["abc"]

def test_replace_folder_and_extension():
    assert pu.replace_folder_and_extension("abc.txt", "tmp", "csv") == "tmp/abc.csv"
    assert pu.replace_folder_and_extension(["abc.txt"], "tmp", "csv") == ["tmp/abc.csv"]
    assert pu.replace_folder_and_extension("abc.txt", None, "csv") == "abc.csv"
    assert pu.replace_folder_and_extension(["abc.txt"], None, "csv") == ["abc.csv"]
    assert pu.replace_folder_and_extension("tmp/abc.txt", None, "csv") == "abc.csv"


def test_read():
    source: str = "./tests/test_data//tranströmer/tranströmer_corpus.zip"
    document_name: str = "tran_2019_01_test.txt"

    apa = pu.read_file_content(zip_or_filename=source, filename=document_name)

    assert apa is not None


def test_read_textfile():
    # utility.read_textfile(filename)
    pass


def test_filename_field_parser():
    # utility.filename_field_parser(meta_fields)
    pass
