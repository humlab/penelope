import itertools
import math
from typing import Callable, Iterable, List, Sequence

import bokeh
import bokeh.models
import bokeh.plotting
import numpy as np
import pandas as pd
import scipy
from bokeh.plotting import figure

from penelope.notebook.utility import generate_temporal_ticks
from penelope.utility import take

DEFAULT_FIGOPTS: dict = dict(width=1000, height=600)
DEFAULT_PALETTE = bokeh.palettes.Category10[10]


def generate_colors(n: int, palette: Sequence[str]) -> Iterable[str]:
    return take(n, itertools.cycle(palette))


def pchip_interpolate_frame(df: pd.DataFrame, step: float = 0.1, columns: List[str] = None) -> pd.DataFrame:

    xs: np.ndarray = np.arange(df.index.min(), df.index.max() + step, step)
    data: dict = {'category': xs}
    columns = columns if columns is not None else df.columns
    for column in columns:
        serie = df[column]
        spliner = scipy.interpolate.PchipInterpolator(df.index, serie)
        data[column] = spliner(xs)

    return pd.DataFrame(data)


def plot_stacked_bar(df: pd.DataFrame, **figopts):

    figopts = {**DEFAULT_FIGOPTS, **(figopts or {})}

    columns: List[str] = df.columns.tolist()

    data_source: dict = dict(category=[str(x) for x in df.index], **{column: df[column] for column in columns})
    colors: Iterable[str] = generate_colors(len(columns), DEFAULT_PALETTE)

    p: figure = figure(x_range=data_source['category'], **figopts, sizing_mode='scale_width')

    p.left[0].formatter.use_scientific = False  # pylint: disable=unsubscriptable-object

    p.vbar_stack(columns, x='category', color=colors, width=0.9, source=data_source, legend_label=columns)

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"

    return p


def to_multiline_data_source(data: pd.DataFrame, smoother: Callable = None) -> dict:
    """Compile multiline plot data for token ids `indices`, optionally applying `smoothers` functions"""

    columns: List[str] = data.columns.tolist()
    colors: Iterable[str] = generate_colors(len(columns), DEFAULT_PALETTE)

    xs_j, ys_js = data.index, [data[column] for column in columns]

    if smoother is not None:
        xs_js, ys_js = list(zip(*[smoother(xs_j, ys_j) for ys_j in ys_js]))
    else:
        xs_js = [xs_j] * len(ys_js)

    data_source = {'xs': xs_js, 'ys': ys_js, 'label': columns, 'color': colors}

    return data_source


def plot_multiline(*, df: pd.DataFrame, smooth: bool = False, **figopts) -> figure:

    x_ticks: Sequence[int] = None

    figopts = {**DEFAULT_FIGOPTS, **(figopts or {})}

    if smooth:
        df = pchip_interpolate_frame(df).set_index('category') if smooth else df
        x_ticks = generate_temporal_ticks(df.index.tolist())

    data_source: dict = to_multiline_data_source(data=df, smoother=None)

    p: figure = figure(**(figopts or {}), sizing_mode='scale_width')

    p.left[0].formatter.use_scientific = False  # pylint: disable=unsubscriptable-object
    p.y_range.start = 0
    p.yaxis.axis_label = 'Frequency'
    p.toolbar.autohide = True

    if x_ticks is not None:
        p.xaxis.ticker = x_ticks

    p.xaxis.major_label_orientation = math.pi / 4
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    p.xgrid.band_fill_alpha = 0.01
    p.xgrid.band_fill_color = "black"

    _ = p.multi_line(xs='xs', ys='ys', legend_field='label', line_color='color', source=data_source)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.0

    return p
