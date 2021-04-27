import warnings

import bokeh
import bokeh.plotting
import bokeh.transform
import penelope.utility as utility
from IPython.display import display

from .. import ipyaggrid_utility, widgets_utils

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = utility.get_logger()


def _setup_glyph_coloring(_, color_high=0.3):

    # colors = list(reversed(bokeh.palettes.Greens[9]))
    colors = [
        '#ffffff',
        '#f7fcf5',
        '#e5f5e0',
        '#c7e9c0',
        '#a1d99b',
        '#74c476',
        '#41ab5d',
        '#238b45',
        '#006d2c',
        '#00441b',
    ]
    mapper = bokeh.models.LinearColorMapper(palette=colors, low=0.0, high=color_high)
    color_transform = bokeh.transform.transform('weight', mapper)
    color_bar = bokeh.models.ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        ticker=bokeh.models.BasicTicker(desired_num_ticks=len(colors)),
        formatter=bokeh.models.PrintfTickFormatter(format=" %5.2f"),
    )
    return color_transform, color_bar


def compute_int_range_categories(values):
    categories = values.unique()
    if all(map(utility.isint, categories)):
        # values = [ int(v) for v in values]
        categories = [str(x) for x in sorted([int(y) for y in categories])]
        return categories
    return sorted(list(categories))


HEATMAP_FIGOPTS = dict(title="Topic heatmap", toolbar_location="right", x_axis_location="above", plot_width=1200)


def plot_topic_relevance_by_year(
    df, xs, ys, flip_axis, titles, text_id, **figopts
):  # pylint: disable=too-many-arguments, too-many-locals

    line_height = 7
    if flip_axis is True:
        xs, ys = ys, xs
        line_height = 10

    # import holoviews as hv
    # from holoviews import opts

    # heatmap = hv.HeatMap((df[xs], df[ys], np.random.randn(100), np.random.randn(100)), vdims=['z', 'z2']).redim.range(z=(-2, 2))
    # heatmap.opts(opts.HeatMap(tools=['hover'], colorbar=True, width=325, toolbar='above'))

    # return heatmap

    x_range = compute_int_range_categories(df[xs])
    y_range = compute_int_range_categories(df[ys])

    color_high = max(df.weight.max(), 0.3)
    color_transform, color_bar = _setup_glyph_coloring(df, color_high=color_high)

    source = bokeh.models.ColumnDataSource(df)

    if x_range is not None:
        figopts['x_range'] = x_range

    if y_range is not None:
        figopts['y_range'] = y_range
        figopts['plot_height'] = max(len(y_range) * line_height, 600)

    p = bokeh.plotting.figure(**figopts)

    args = dict(x=xs, y=ys, source=source, alpha=1.0, hover_color='red')

    cr = p.rect(width=1, height=1, line_color=None, fill_color=color_transform, **args)

    p.x_range.range_padding = 0
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0
    p.add_layout(color_bar, 'right')

    p.add_tools(
        bokeh.models.HoverTool(
            tooltips=None,
            callback=widgets_utils.glyph_hover_callback2(source, 'topic_id', titles.index, titles, text_id),
            renderers=[cr],
        )
    )
    return p


def display_heatmap(
    weights, titles, key='max', flip_axis=False, glyph='Circle', aggregate=None, output_format=None
):  # pylint: disable=unused-argument
    try:

        ''' Display aggregate value grouped by year  '''
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
                text_id='topic_relevance',
                **HEATMAP_FIGOPTS,
            )

            bokeh.plotting.show(p)

        else:
            g = ipyaggrid_utility.display_grid(weights)
            display(g)

    except Exception as ex:
        # raise
        logger.error(ex)
