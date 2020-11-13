import pandas as pd
from IPython.display import display
from penelope.notebook.ipyaggrid_utility import display_grid

from . import data_compilers

NAME = "Table"

compile = data_compilers.compile_year_token_vector_data  # pylint: disable=redefined-builtin


def setup(container, **_):  # pylint: disable=unused-argument
    pass


def plot(data, **_):  # pylint: disable=unused-argument

    df = pd.DataFrame(data=data)
    df = df[['year'] + [x for x in df.columns if x != 'year']].set_index('year')
    g = display_grid(df)
    display(g)
