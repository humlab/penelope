from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.io import show
from bokeh.models import FuncTickFormatter
from bokeh.plotting import figure
from scipy.interpolate import PchipInterpolator

from penelope.notebook.utility import generate_colors, generate_temporal_ticks


def pchip_interpolate_frame(
    df: pd.DataFrame,
    step: float = 0.1,
    columns: list[str] = None,
) -> pd.DataFrame:
    """Smoothes `columns` in `df` using  Scipy PchipInterpolator."""

    xs: np.ndarray = np.arange(df.index.min(), df.index.max() + step, step)
    columns = columns if columns is not None else df.columns
    data: dict = {column: PchipInterpolator(df.index, df[column])(xs) for column in columns}
    data['category'] = xs
    return pd.DataFrame(data)


def plot_multiple_value_series(
    *,
    kind: str,
    data: pd.DataFrame,
    category_name: str,
    columns: list[str] = None,
    title: str = None,
    fig_opts: dict = None,
    plot_opts: dict = None,
):
    fig_opts = {
        **dict(plot_height=250, sizing_mode='scale_width', title=title, toolbar_location=None, tools=""),
        **(dict(title=title) if title else {}),
        **(fig_opts or {}),
    }
    plot_opts = plot_opts or {}

    columns = columns or [x for x in data.columns if x != category_name]
    category_series = [str(x) for x in data[category_name]] * len(columns)
    value_series = [data[name].values for name in columns]

    p = figure(x_range=category_series[0], **fig_opts)

    if kind == 'vbar_stack':
        plot_opts = {**dict(width=0.2), **plot_opts}
        p.vbar_stack(columns, x=category_name, source=data, **plot_opts)
    else:
        p.multi_line(xs=[data.year] * len(columns), ys=value_series, **plot_opts)

    show(p)


def pchip_interpolate_frame2(
    df: pd.DataFrame, step: float = 0.1, columns: list[str] = None, category_column: str = 'year'
) -> pd.DataFrame:
    """Smoothes `columns` in `df` using  Scipy PchipInterpolator."""

    xs: np.ndarray = np.arange(df.index.min(), df.index.max() + step, step)
    columns = columns if columns is not None else df.columns
    data: dict = {column: PchipInterpolator(df.index, df[column])(xs) for column in columns}
    data[category_column] = xs
    return pd.DataFrame(data)


def plot_multiple_value_series2(
    *,
    kind: str,
    data: pd.DataFrame,
    category_name: str,
    columns: list[str] = None,
    smooth: bool = True,
    fig_opts: dict = None,
    plot_opts: dict = None,
    palette: str | list[str] = "Category10",
    n_tick: int = 5,
):
    fig_opts = {
        **dict(plot_height=400, plot_width=1000, sizing_mode='scale_both'),
        **(fig_opts or {}),
    }  # , x_axis_type=None
    plot_opts = plot_opts or {}

    if smooth and 'line' in kind.lower():
        data = pchip_interpolate_frame(data.set_index(category_name))

    columns = columns or [x for x in data.columns if x != category_name]
    category_series = [[x for x in data[category_name].values]] * len(columns)
    value_series = [data[name].values for name in columns]

    if kind.lower() == 'bar':
        fig_opts['x_range'] = data[category_name].astype(str)

    p = figure(**fig_opts)
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.y_range.start = 0
    p.left[0].formatter.use_scientific = False  # pylint: disable=unsubscriptable-object

    colors: list[str] = generate_colors(len(columns), palette=palette)

    if kind.lower() == 'bar':
        offset = data[category_name].min() % n_tick
        data[category_name] = data[category_name].astype(str)
        p.axis.formatter = FuncTickFormatter(
            code=f"""
            return ((index == 0) || ((index - {offset}) % {n_tick} == 0)) ? tick : "";
        """
        )
        p.vbar_stack(columns, x=category_name, width=0.2, source=data, color=colors, legend_label=columns, **plot_opts)
    else:
        p.xaxis.ticker = generate_temporal_ticks(category_series[0])
        line_source = dict(xs=category_series, ys=value_series, color=colors, legend=columns)
        p.multi_line(xs='xs', ys='ys', color='color', legend='legend', source=line_source, line_width=1, **plot_opts)

    p.legend.location = "top_right"
    p.legend.orientation = "vertical"

    show(p)
