import warnings
from typing import List, Sequence

import bokeh
import bokeh.plotting
import bokeh.transform
import pandas as pd
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    HoverTool,
    LinearColorMapper,
    PrintfTickFormatter,
)
from IPython.display import display
from loguru import logger

from penelope import utility as pu

from .. import grid_utility as gu
from .. import widgets_utils as wu

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

COLORS = ['#ffffff', '#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
HEATMAP_FIGOPTS = dict(title="Topic heatmap", toolbar_location="right", x_axis_location="above", width=1200)


def _setup_glyph_coloring(_, color_high=0.3):

    # colors = list(reversed(bokeh.palettes.Greens[9]))

    mapper = LinearColorMapper(palette=COLORS, low=0.0, high=color_high)
    color_transform = bokeh.transform.transform('weight', mapper)
    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=len(COLORS)),
        formatter=PrintfTickFormatter(format=" %5.2f"),
    )
    return color_transform, color_bar


def to_categories(values: pd.Series) -> List[str]:
    """Make unique and sorted string categories."""

    categories: Sequence[int] = values.unique()

    if all(pu.isint(x) for x in categories):
        return [str(x) for x in sorted([int(y) for y in categories])]

    return sorted(list(categories))


def plot_topic_relevance_by_year(
    weights: pd.DataFrame,
    xs: Sequence[int],
    ys: Sequence[int],
    flip_axis: bool,
    titles: pd.Series,
    element_id: str,
    **figopts,
):  # pylint: disable=too-many-arguments, too-many-locals

    line_height: int = 7
    if flip_axis:
        xs, ys = ys, xs
        line_height = 10

    x_range: List[str] = to_categories(weights[xs])
    y_range: List[str] = to_categories(weights[ys])

    color_high = max(weights.weight.max(), 0.3)
    color_transform, color_bar = _setup_glyph_coloring(weights, color_high=color_high)

    source: ColumnDataSource = ColumnDataSource(weights)

    if x_range is not None:
        figopts['x_range'] = x_range

    if y_range is not None:
        figopts['y_range'] = y_range
        figopts['height'] = max(len(y_range) * line_height, 600)

    p = bokeh.plotting.figure(**figopts, sizing_mode='scale_width')

    cr = p.rect(
        x=xs,
        y=ys,
        source=source,
        alpha=1.0,
        hover_color='red',
        width=1,
        height=1,
        line_color=None,
        fill_color=color_transform,
    )

    p.x_range.range_padding = 0
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0
    p.add_layout(color_bar, 'right')

    text_source: ColumnDataSource = ColumnDataSource(dict(text_id=titles.index.tolist(), text=titles.tolist()))

    code: str = wu.display_text_on_hover_js_code(
        element_id=element_id, id_name='topic_id', text_name='text', glyph_name='glyph', glyph_data='glyph_data'
    )
    callback: CustomJS = CustomJS(args={'glyph': cr.data_source, 'glyph_data': text_source}, code=code)

    p.add_tools(HoverTool(tooltips=None, callback=callback, renderers=[cr]))
    return p


def display_heatmap(
    weights: pd.DataFrame,
    titles: pd.DataFrame,
    key: str = 'max',  # pylint: disable=unused-argument
    flip_axis: bool = False,
    glyph: str = 'Circle',  # pylint: disable=unused-argument
    aggregate: str = None,
    output_format: str = None,
):
    '''Display aggregate value grouped by year'''
    try:

        weights['weight'] = weights[aggregate]
        weights['year'] = weights.year.astype(str)
        weights['topic_id'] = weights.topic_id.astype(str)

        if len(weights) == 0:
            print("No data! Please change selection.")
            return

        if output_format.lower() == 'heatmap':

            p = plot_topic_relevance_by_year(
                weights,
                xs='year',
                ys='topic_id',
                flip_axis=flip_axis,
                titles=titles,
                element_id='topic_relevance',
                **HEATMAP_FIGOPTS,
            )

            bokeh.plotting.show(p)

        elif output_format.lower() in ('xlsx', 'csv', 'clipboard'):
            pu.ts_store(data=weights, extension=output_format.lower(), basename='heatmap_weights')
        else:
            g = gu.table_widget(weights)
            display(g)

    except Exception as ex:
        # raise
        logger.error(ex)
