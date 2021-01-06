import os

import numpy as np
import pandas as pd
import pytest  # pylint: disable=unused-import
from penelope.utility import (
    create_iterator,
    extract_filename_metadata,
    filename_satisfied_by,
    list_filenames,
    namelist,
    pandas_read_csv_zip,
    pandas_to_csv_zip,
    strip_path_and_add_counter,
    strip_path_and_extension,
    strip_paths,
)
from penelope.utility.filename_utils import strip_extensions
from tests.utils import TEST_CORPUS_FILENAME

OUTPUT_FOLDER = './tests/output'


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


def test_create_iterator():
    stream = create_iterator(TEST_CORPUS_FILENAME, ['dikt_2019_01_test.txt'], filename_pattern='*.txt', as_binary=False)
    assert len([x for x in stream]) == 1


def test_list_filenames_when_source_is_a_filename():

    filenames = list_filenames(TEST_CORPUS_FILENAME, filename_pattern='*.txt', filename_filter=None)
    assert len(filenames) == 5

    filenames = list_filenames(TEST_CORPUS_FILENAME, filename_pattern='*.dat', filename_filter=None)
    assert len(filenames) == 0

    filenames = list_filenames(
        TEST_CORPUS_FILENAME, filename_pattern='*.txt', filename_filter=['dikt_2019_01_test.txt']
    )
    assert len(filenames) == 1

    filenames = list_filenames(
        TEST_CORPUS_FILENAME, filename_pattern='*.txt', filename_filter=lambda x: x == 'dikt_2019_01_test.txt'
    )
    assert len(filenames) == 1


def list_filenames_when_source_is_folder():
    # utility.list_filenames(folder_or_zip, filename_pattern: str="*.txt", filename_filter: Union[List[str],Callable]=None)
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


def create_pandas_test_data():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]}, index=[4, 5])
    df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'], index=[1, 2, 3])
    data = [(df1, 'df1.csv'), (df2, 'df2.csv')]
    return data


def test_pandas_to_csv_zip():

    filename = os.path.join(OUTPUT_FOLDER, "test_pandas_to_csv_zip.zip")
    data = create_pandas_test_data()

    pandas_to_csv_zip(filename, dfs=data, extension='csv', sep='\t')

    assert os.path.isfile(filename)
    assert set(namelist(filename, pattern="*.csv")) == set({'df1.csv', 'df2.csv'})


def test_pandas_read_csv_zip():

    filename = os.path.join(OUTPUT_FOLDER, "test_pandas_to_csv_zip.zip")
    expected_data = create_pandas_test_data()
    pandas_to_csv_zip(filename, dfs=expected_data, extension='csv', sep='\t')

    data = pandas_read_csv_zip(filename, pattern='*.csv', sep='\t', index_col=0)

    assert 'df1.csv' in data and 'df2.csv' in data
    assert ((data['df1.csv'] == expected_data[0][0]).all()).all()
    assert ((data['df2.csv'] == expected_data[1][0]).all()).all()
