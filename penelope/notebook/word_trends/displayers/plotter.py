import itertools
import math
from typing import Any, Callable, List, Sequence, Tuple, Union

import bokeh
import numpy as np
from bokeh.plotting import figure

import penelope.common.curve_fit as cf
from penelope.corpus import VectorizedCorpus
from penelope.corpus.document_index import DocumentIndexHelper
from penelope.utility import get_logger

logger = get_logger()
SmootherFunction = Callable[[Any, Any], Tuple[Any, Any]]


def yearly_token_distribution_single_line_plot(
    xs: Union[str, Sequence[float]],
    ys: Union[str, Sequence[float]],
    *,
    plot: figure = None,
    title: str = '',
    color: str = 'navy',
    ticker_labels=None,
    smoothers: Sequence[SmootherFunction] = None,
    **kwargs,
) -> figure:
    """Plots a distribution defined by coordinate vectors `xs` and `ys`.

    Parameters
    ----------
    xs : Union[str, Sequence[float]]
        X-coordinates
    ys : Union[str, Sequence[float]]
        Y-coordinates
    plot : figure, optional
        figure to use, if None then a new figure is created, by default None
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
    figure
    """
    if plot is None:

        p: figure = bokeh.plotting.figure(
            width=kwargs.get('width', 400),
            height=kwargs.get('height', 200),
            sizing_mode='scale_width',
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
) -> figure:
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
    figure
        bokeh figure
    """
    # x_corpus = x_corpus.todense()

    smoothers = smoothers or [cf.rolling_average_smoother('nearest', 3), cf.pchip_spline]

    colors = itertools.cycle(bokeh.palettes.Category10[10])
    x_range = DocumentIndexHelper.year_range(x_corpus.document_index)
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
                width=width if n_columns is None else max(int(width / n_columns), 400),
                height=height if n_columns is None else max(int(height / n_columns), 300),
                ticker_labels=xs if n_columns is None else None,
                smoothers=smoothers,
            )
            plots.append(p)

        except Exception as ex:  # pylint: disable=bare-except

            logger.exception(ex)

    if n_columns is not None:

        p = bokeh.layouts.gridplot([plots[u : u + n_columns] for u in range(0, len(indices), n_columns)])

    return p
