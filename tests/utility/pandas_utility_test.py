import pandas as pd
from penelope.utility import try_split_column


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
