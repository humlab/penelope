from __future__ import annotations

import math
from itertools import cycle, islice
from typing import Callable, Iterable, List, Sequence

import bokeh
import bokeh.models as bm
import bokeh.plotting as bp
import numpy as np
import pandas as pd
from bokeh.palettes import all_palettes
from scipy.interpolate import PchipInterpolator

DEFAULT_FIGOPTS: dict = dict(width=1000, height=600)
DEFAULT_PALETTE = bokeh.palettes.Category10[10]


def high_bound(categories: list[int], n_tick: int) -> tuple[int, int]:
    return (lambda x: x if x % n_tick == 0 else x + (n_tick - x % n_tick))(int(max(categories)))


def low_bound(categories: list[int], n_tick: int) -> int:
    return (lambda x: x - (x % n_tick))(int(min(categories)))


def generate_colors(n: int, palette: Iterable[str] | str = 'Category20', palette_id: int = None) -> Iterable[str]:

    if not isinstance(palette, str):
        # return take(n, cycle(palette))
        return list(islice(cycle(palette), n))

    if palette in all_palettes:
        palette_id: int = palette_id if palette_id is not None else max(all_palettes[palette].keys())
        return list(islice(cycle(all_palettes[palette][palette_id]), n))

    raise ValueError(f"unknown palette {palette}")


def generate_temporal_ticks(categories: list[int], n_tick: int = 5) -> list[int]:
    """Gets ticks every n_tick years if category is year
    Returns all categories if all values are either, lustrum and decade"""

    if all(int(x) % 5 in (0, 5) for x in categories):
        return categories

    return list(range(low_bound(categories, n_tick), high_bound(categories, n_tick) + 1, n_tick))


def pchip_interpolate_frame(
    df: pd.DataFrame, step: float = 0.1, columns: list[str] = None, category_name: str = 'category'
) -> pd.DataFrame:
    """Smoothes `columns` in `df` using  Scipy PchipInterpolator."""

    xs: np.ndarray = np.arange(df.index.min(), df.index.max() + step, step)
    columns = columns if columns is not None else df.columns
    data: dict = {column: PchipInterpolator(df.index, df[column])(xs) for column in columns}
    data[category_name] = xs
    return pd.DataFrame(data)


def plot_multiple_value_series(
    *,
    kind: str,
    data: pd.DataFrame,
    category_name: str,
    columns: list[str] = None,
    smooth: bool = True,
    fig_opts: dict = None,
    plot_opts: dict = None,
    palette: str | list[str] = "Category10",
    colors: list[str] = None,
    n_tick: int = 5,
):
    fig_opts = {
        **dict(height=400, width=1000, sizing_mode='scale_both'),
        **(fig_opts or {}),
    }  # , x_axis_type=None
    plot_opts = plot_opts or {}

    if smooth and 'line' in kind.lower():
        data = pchip_interpolate_frame(data.set_index(category_name), category_name=category_name)

    columns = columns or [x for x in data.columns if x != category_name]
    category_series = [[x for x in data[category_name].values]] * len(columns)
    value_series = [data[name].values for name in columns]

    if kind.lower() == 'bar':
        fig_opts['x_range'] = data[category_name].astype(str)

    p = bp.figure(**fig_opts)
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.y_range.start = 0
    p.left[0].formatter.use_scientific = False  # pylint: disable=unsubscriptable-object

    colors: list[str] = colors or generate_colors(len(columns), palette=palette)

    if kind.lower() == 'bar':
        offset = data[category_name].min() % n_tick
        data[category_name] = data[category_name].astype(str)
        p.axis.formatter = bm.FuncTickFormatter(
            code=f"""
            return ((index == 0) || ((index - {offset}) % {n_tick} == 0)) ? tick : "";
        """
        )
        p.vbar_stack(columns, x=category_name, width=0.2, source=data, color=colors, legend_label=columns, **plot_opts)
    else:
        p.xaxis.ticker = generate_temporal_ticks(category_series[0])
        line_source = dict(xs=category_series, ys=value_series, color=colors, legend=columns)
        p.multi_line(
            xs='xs', ys='ys', color='color', legend_field='legend', source=line_source, line_width=1, **plot_opts
        )

    p.legend.location = "top_right"
    p.legend.orientation = "vertical"

    bp.show(p)


def plot_stacked_bar(df: pd.DataFrame, **figopts):

    figopts = {**DEFAULT_FIGOPTS, **(figopts or {})}

    columns: List[str] = df.columns.tolist()

    data_source: dict = dict(category=[str(x) for x in df.index], **{column: df[column] for column in columns})
    colors: Iterable[str] = generate_colors(len(columns), DEFAULT_PALETTE)

    p: bp.figure = bp.figure(x_range=data_source['category'], **figopts, sizing_mode='scale_width')

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


def _to_multiline_data_source(data: pd.DataFrame, smoother: Callable = None) -> dict:
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


def plot_multiline(*, df: pd.DataFrame, smooth: bool = False, **figopts) -> bp.figure:

    x_ticks: Sequence[int] = None

    figopts = {**DEFAULT_FIGOPTS, **(figopts or {})}

    if smooth:
        df = pchip_interpolate_frame(df, category_name='category').set_index('category') if smooth else df
        x_ticks = generate_temporal_ticks(df.index.tolist())

    data_source: dict = _to_multiline_data_source(data=df, smoother=None)

    p: bp.figure = bp.figure(**(figopts or {}), sizing_mode='scale_width')

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
