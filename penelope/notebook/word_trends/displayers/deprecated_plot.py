import itertools
import math
from typing import Any, Callable, List, Mapping, Sequence, Tuple, Union

import bokeh
import numpy as np
import penelope.common.curve_fit as cf
from bokeh.plotting import Figure
from penelope.corpus import VectorizedCorpus
from penelope.utility import get_logger, take

logger = get_logger()
SmootherFunction = Callable[[Any, Any], Tuple[Any, Any]]


def yearly_token_distributions_datasource(x_corpus: VectorizedCorpus, indices: List[int], *_):
    """Returns a dictionary containing distributions sliced from `x_corpus` for tokens identities
    found in `indices`. The corresponding `token` given by `id2token` is used as key.
    The dictionary also contains the `year` vector common for all distrbutions.

    Parameters
    ----------
    x_corpus : VectorizedCorpus
        [description]
    indices : List[int]
        List of token indices (token ids)

    Returns
    -------
    Mapping
    """
    xs = x_corpus.xs_years()
    data = {x_corpus.id2token[token_id]: x_corpus.bag_term_matrix[:, token_id] for token_id in indices}
    data["year"] = xs

    return data


def yearly_token_distributions_multiline_datasource(
    x_corpus: VectorizedCorpus, indices: List[int], smoothers: Callable = None, palette: Any = None
) -> Mapping:
    """Returns a dictionary containing token distributions sliced from `x_corpus` using indices `indices`.
        The data is prepared for a bokeh `multiline` plot which requires the distribution data
        to be stored as two lists `xs` and `ys` containing coordinates (i.e. a list) for each line.

        The i:th elements in `xs` and `ys` specifies the distribution for token `i` in index `indicies`

        The `year` vectors in `xs` are the same for all distributions.

             xs = [ [ years ], [ years ], ..., [ years ]]
             ys = [ [ dist1 ], [ dist2 ], ..., [ distn ]]
          token = [  "token1",  "token2", ...,  "token" ]

        Optionally, a list of functions can be supplied to smooth the data.

    Parameters
    ----------
    x_corpus : VectorizedCorpus
    indices : List[int]
        List of token indices (token ids)
    smoothers : Callable, optional
        List of smoothing functions, by default None
    palette : [type], optional
        Color palette, by default None

    Returns
    -------
    Mapping
    """
    xs = x_corpus.xs_years()

    if len(smoothers or []) > 0:
        xs_data = []
        ys_data = []
        for j in indices:
            xs_j = xs
            ys_j = x_corpus.bag_term_matrix[:, j]
            for smoother in smoothers:
                xs_j, ys_j = smoother(xs_j, ys_j)
            xs_data.append(xs_j)
            ys_data.append(ys_j)
    else:
        xs_data = [xs.tolist()] * len(indices)
        ys_data = [x_corpus.bag_term_matrix[:, token_id].tolist() for token_id in indices]

    palette = palette or bokeh.palettes.Category10[10]

    data = {
        "xs": xs_data,
        "ys": ys_data,
        "token": [x_corpus.id2token[token_id].upper() for token_id in indices],
        "color": take(len(indices), itertools.cycle(palette)),
    }
    return data


def yearly_token_distributions_bar_plot(data: Mapping, **_):

    years = [str(y) for y in data["year"]]

    data["year"] = years

    tokens = [w for w in data.keys() if w != "year"]

    source = bokeh.models.ColumnDataSource(data=data)

    max_value = max([max(data[key]) for key in data if key != "year"]) + 0.005

    p = bokeh.plotting.figure(
        x_range=years,
        y_range=(0, max_value),
        plot_height=400,
        plot_width=1000,
        title="Word frequecy by year",
    )

    colors = itertools.islice(itertools.cycle(bokeh.palettes.d3["Category20b"][20]), len(tokens))

    offset = -0.25
    v = []
    for token in tokens:
        w = p.vbar(
            x=bokeh.transform.dodge("year", offset, range=p.x_range),
            top=token,
            width=0.2,
            source=source,
            color=next(colors),
        )  # , legend_label=token)
        offset += 0.25
        v.append(w)

    p.x_range.range_padding = 0.04
    p.xaxis.major_label_orientation = math.pi / 4
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Note: Fixed Bokeh legend error
    # p.legend.location = "top_right"
    # p.legend.orientation = "vertical"

    legend = bokeh.models.Legend(items=[(x, [v[i]]) for i, x in enumerate(tokens)])
    p.add_layout(legend, "left")

    return p


def empty_multiline_datasource():

    data = {
        "xs": [[0]],
        "ys": [[0]],
        "label": [""],
        "color": ["red"],
    }

    return bokeh.models.ColumnDataSource(data)


def yearly_token_distributions_multiline_plot(
    data_source, *, x_ticks=None, plot_width: int = 1000, plot_height: int = 800, **_
):

    p = bokeh.plotting.figure(plot_width=plot_width, plot_height=plot_height)
    p.y_range.start = 0
    p.yaxis.axis_label = "Frequency"
    p.toolbar.autohide = True

    if x_ticks is not None:
        p.xaxis.ticker = x_ticks

    p.xaxis.major_label_orientation = math.pi / 4
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    _ = p.multi_line(xs="xs", ys="ys", legend_field="label", line_color="color", source=data_source)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.0

    return p


def yearly_token_distribution_single_line_plot(
    xs: Union[str, Sequence[float]],
    ys: Union[str, Sequence[float]],
    *,
    plot: Figure = None,
    title: str = '',
    color: str = 'navy',
    ticker_labels=None,
    smoothers: Sequence[SmootherFunction] = None,
    **kwargs,
) -> Figure:
    """Plots a distribution defined by coordinate vectors `xs` and `ys`.

    Parameters
    ----------
    xs : Union[str, Sequence[float]]
        X-coordinates
    ys : Union[str, Sequence[float]]
        Y-coordinates
    plot : Figure, optional
        Figure to use, if None then a new Figure is created, by default None
    title : str, optional
        Plot title, by default ''
    color : str, optional
        Color for plot, by default 'navy'
    ticker_labels : [type], optional
        Ticker labels to use, by default None
    smoothers : Sequence[SmootherFunction], optional
        List of functions to apply on coordinates before plot, by default None

    Returns
    -------
    Figure
    """
    if plot is None:

        p: Figure = bokeh.plotting.figure(
            plot_width=kwargs.get('plot_width', 400), plot_height=kwargs.get('plot_height', 200)
        )
        p.y_range.start = 0
        # p.y_range.end = 0.5
        # p.title.text = title.upper()
        p.yaxis.axis_label = 'Frequency'
        p.toolbar.autohide = True
        if ticker_labels is not None:
            p.xaxis.ticker = ticker_labels
        p.xaxis.major_label_orientation = math.pi / 2
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

    else:
        p = plot

    _ = p.scatter(xs, ys, size=2, color=color, alpha=1.0, legend_label=title)

    # p.line(xs, ys , line_width=2, color=color, alpha=0.5, legend_label=title)
    for smoother in smoothers or []:
        xs, ys = smoother(xs, ys)

    _ = p.line(xs, ys, line_width=2, color=color, alpha=0.5, legend_label=title)

    # p.vbar(x=xs, top=ys, width=0.5, color=color, alpha=0.1)
    # p.step(xs, ys, line_width=2, color=color, alpha=0.5)
    # _, _, _, lx, ly = gof.fit_ordinary_least_square(ys, xs)
    # p.line(x=lx, y=ly, line_width=1, color=color, alpha=0.6, legend_label=title)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.0

    return p


def yearly_token_distribution_multiple_line_plot(
    x_corpus: VectorizedCorpus,
    indices: List[int],
    n_columns: int = 3,
    *,
    width: int = 1000,
    height: int = 600,
    smoothers: SmootherFunction = None,
) -> Figure:
    """Plots word distributions (columns) over time for words at indices `indices` in `x_corpus`' (i.e. the document-term-matrix.)

    The `x-axis` is defined by the range of years that the corpus spans.

    If `n_columns` is None then all distributions are plotted on the same plot. If `n_columns` is specified, then
    the distributions are plotted on separate plot in a grid with 'n_columns' plots per row.,

    Parameters
    ----------
    x_corpus : VectorizedCorpus
    indices : List[int]
        List of indices (token ids) to plot.
    n_columns : int, optional
        Number of plots per row, if `None` then all in one plot, by default 3
    width : int, optional
        Total plot width, by default 1000
    height : int, optional
        Total plot height, by default 600
    smoothers : SmootherFunction, optional
        List of functions  to use to smooth the lines, by default None

    Returns
    -------
    Figure
        bokeh Figure
    """
    # x_corpus = x_corpus.todense()

    smoothers = smoothers or [cf.rolling_average_smoother('nearest', 3), cf.pchip_spline]

    colors = itertools.cycle(bokeh.palettes.Category10[10])
    x_range = x_corpus.year_range()
    xs = np.arange(x_range[0], x_range[1] + 1, 1)

    plots = []
    p = None

    for token_id in indices:
        try:

            # ys = x_corpus.data[:, token_id]
            ys = x_corpus.data.getcol(token_id).A.ravel()
            p = yearly_token_distribution_single_line_plot(
                xs,
                ys,
                plot=p if n_columns is None else None,
                title=x_corpus.id2token[token_id].upper(),
                color=next(colors),
                plot_width=width if n_columns is None else max(int(width / n_columns), 400),
                plot_height=height if n_columns is None else max(int(height / n_columns), 300),
                ticker_labels=xs if n_columns is None else None,
                smoothers=smoothers,
            )
            plots.append(p)

        except Exception as ex:  # pylint: disable=bare-except

            logger.exception(ex)

    if n_columns is not None:

        p = bokeh.layouts.gridplot([plots[u : u + n_columns] for u in range(0, len(indices), n_columns)])

    return p


# def plot_distribution2(ys: Union[str, Sequence[float]], window_size, mode='nearest'):

#     xs = np.arange(0, len(ys), 1)
#     yw = scipy.ndimage.filters.uniform_filter1d(ys, size=window_size, mode=mode)
#     xw = np.arange(0, len(yw), 1)

#     # xp, yp = cf.fit_curve(cf.fit_polynomial3, xs, ys, step=0.1)
#     xp, yp = cf.fit_polynomial4(xs, ys)
#     p = bokeh.plotting.figure(
#         plot_width=800,
#         plot_height=600,
#     )
#     p.scatter(xs, ys, size=6, color='red', alpha=0.6, legend_label='actual')
#     p.line(x=xs, y=ys, line_width=1, color='red', alpha=0.6, legend_label='actual')
#     p.line(x=xp, y=yp, line_width=1, color='green', alpha=0.6, legend_label='poly')
#     p.xgrid.grid_line_color = None
#     p.ygrid.grid_line_color = None

#     p.line(x=xw, y=yw, line_width=1, color='green', alpha=1, legend_label='rolling')

#     return p
