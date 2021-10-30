import os

import numpy as np
import pandas as pd
from penelope.utility import list_filenames, pandas_read_csv_zip, pandas_to_csv_zip, try_split_column
from penelope.utility.pandas_utils import create_mask
from tests.utils import OUTPUT_FOLDER


def test_try_split_column():

    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    df = try_split_column(df, 'A', '/', ['x', 'y'], False, 1)
    assert df.columns.to_list() == ['A']

    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    df = try_split_column(df, 'B', '/', ['x', 'y'], False, 1)
    assert df.columns.to_list() == ['A']

    df = pd.DataFrame({'A': ['a/b', 'b/c', 'c']})
    df = try_split_column(df, 'A', '/', ['x', 'y'], False, 3)
    assert df.columns.to_list() == ['A']

    df = pd.DataFrame({'A': ['a/b', 'b/c', 'c']})
    df = try_split_column(df, 'A', '/', ['x', 'y'], False, 2)
    assert df.x.tolist() == ['a', 'b', 'c']
    assert df.y.tolist() == ['b', 'c', None]

    df = pd.DataFrame({'A': ['a/b', 'b/c', 'c/d']})
    df = try_split_column(df, 'A', '/', ['x', 'y'], False, 2)
    assert df.columns.to_list() == ['A', 'x', 'y']


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
    assert set(list_filenames(zip_or_filename=filename, filename_pattern="*.csv")) == set({'df1.csv', 'df2.csv'})


def test_pandas_read_csv_zip():

    filename: str = os.path.join(OUTPUT_FOLDER, "test_pandas_to_csv_zip.zip")
    expected_data = create_pandas_test_data()
    pandas_to_csv_zip(filename, dfs=expected_data, extension='csv', sep='\t')

    data = pandas_read_csv_zip(filename, pattern='*.csv', sep='\t', index_col=0)

    assert 'df1.csv' in data and 'df2.csv' in data
    assert ((data['df1.csv'] == expected_data[0][0]).all()).all()
    assert ((data['df2.csv'] == expected_data[1][0]).all()).all()


def test_create_mask():

    df: pd.DataFrame = pd.DataFrame({'A': ['a', 'a', 'c', 'd'], 'B': [True, False, True, True], 'C': [1, 2, 3, 4]})

    mask: np.ndarray = create_mask(df, {})
    assert len(mask) == 4 and all(mask)

    empty: pd.DataFrame = df[df.A == 'X']
    mask: np.ndarray = create_mask(empty, {'A': 'a'})
    assert len(mask) == 0

    assert (create_mask(df, {'A': 'a'}) == [True, True, False, False]).all()
    assert (create_mask(df, {'A': (True, 'a')}) == [True, True, False, False]).all()
    assert (create_mask(df, {'A': (False, 'a')}) == [False, False, True, True]).all()
    assert (create_mask(df, {'A': (False, ['c', 'd'])}) == [True, True, False, False]).all()
    assert (create_mask(df, {'A': 'a', 'B': True}) == [True, False, False, False]).all()
    assert (create_mask(df, {'A': 'a', 'B': True, 'C': 2}) == [False, False, False, False]).all()
