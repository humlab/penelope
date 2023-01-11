from __future__ import annotations

import math

import bokeh.plotting as bp
import pandas as pd


def display_topic_trends(
    weight_over_time: pd.DataFrame,
    year_range: tuple[int, int],
    value_column: str,
    category_column: str = 'year',
    x_label: str = None,
    y_label: str = None,
    **figopts,
):

    xs: pd.Series = weight_over_time[category_column].astype(str)
    ys: pd.Series = weight_over_time[value_column]

    default_figopts: dict = dict(
        width=1000,
        height=400,
        title='',
        toolbar_location="right",
        x_range=list(map(str, range(year_range[0], year_range[1] + 1))),
        y_range=(0.0, ys.max()),
    )

    figopts: dict = {**default_figopts, **figopts}

    p: bp.figure = bp.figure(**figopts, sizing_mode='scale_width')

    _ = p.vbar(x=xs, top=ys, width=0.5, fill_color="#b3de69")

    p.xaxis.major_label_orientation = math.pi / 4
    p.xgrid.grid_line_color = None
    p.left[0].formatter.use_scientific = False  # pylint: disable=unsubscriptable-object
    p.xaxis[0].axis_label = (x_label or category_column.title().replace('_', ' ')).title()
    p.yaxis[0].axis_label = (y_label or value_column.title().replace('_', ' ')).title()
    p.y_range.start = 0.0
    p.x_range.range_padding = 0.01

    bp.show(p)
