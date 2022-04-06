import numpy as np
import pandas as pd
from bokeh.io import show
from bokeh.plotting import figure
from scipy.interpolate import PchipInterpolator


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
        **dict(plot_height=250, title=title, toolbar_location=None, tools=""),
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
