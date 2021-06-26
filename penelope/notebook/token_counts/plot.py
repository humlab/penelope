import itertools
import math
from typing import Callable, Sequence

import bokeh
import bokeh.models
import bokeh.plotting
import numpy as np
import pandas as pd
import scipy
from penelope.notebook.word_trends.displayers.utils import get_year_category_ticks
from penelope.utility import take

# pd.set_option('plotting.backend', 'pandas_bokeh')
# pd.plotting.output_notebook()


def pchip_interpolate_frame(df: pd.DataFrame):
    x_new = np.arange(df.index.min(), df.index.max() + 0.1, 0.1)
    data = {'category': x_new}
    for column in df.columns:
        serie = df[column]
        spliner = scipy.interpolate.PchipInterpolator(df.index, serie)
        data[column] = spliner(x_new)

    return pd.DataFrame(data)


def plot_by_bokeh(*, data_source: pd.DataFrame, smooth: bool):
    xticks = get_year_category_ticks(data_source.index.tolist())
    data_source = pchip_interpolate_frame(data_source).set_index('category') if smooth else data_source
    return plot_dataframe(data_frame=data_source, x_ticks=xticks, figopts=dict(plot_width=1000, plot_height=600))


def plot_dataframe(
    *, data_frame: pd.DataFrame, x_ticks: Sequence[int] = None, smoother: Callable = None, figopts: dict = None
):
    def data_frame_to_data_source(data: pd.DataFrame, smoother: Callable = None) -> dict:
        """Compile multiline plot data for token ids `indices`, optionally applying `smoothers` functions"""

        columns = data.columns.tolist()

        xs_j = data.index
        ys_js = [data[x] for x in columns]

        if smoother is not None:
            xs_js, ys_js = list(zip(*[smoother(xs_j, ys_j) for ys_j in ys_js]))
        else:
            xs_js = [xs_j] * len(ys_js)

        data_source = {
            'xs': xs_js,
            'ys': ys_js,
            'label': columns,
            'color': take(len(columns), itertools.cycle(bokeh.palettes.Category10[10])),
        }

        return data_source

    data_source = data_frame_to_data_source(data=data_frame, smoother=smoother)
    p = bokeh.plotting.figure(**(figopts or {}))

    # p.sizing_mode = 'scale_width'
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

    # bokeh.plotting.show(p)

    return p
