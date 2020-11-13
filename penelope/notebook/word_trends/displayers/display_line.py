import math

import bokeh

from . import data_compilers

NAME = "Line"

compile = data_compilers.compile_multiline_data  # pylint: disable=redefined-builtin


def setup(container, **kwargs):

    x_ticks = kwargs.get('x_ticks', None)
    plot_width = kwargs.get('plot_width', 1000)
    plot_height = kwargs.get('plot_height', 800)

    data = {'xs': [[0]], 'ys': [[0]], 'label': [""], 'color': ['red']}  # , 'token_id': [ 0 ] }

    data_source = bokeh.models.ColumnDataSource(data)

    p = bokeh.plotting.figure(plot_width=plot_width, plot_height=plot_height)
    p.y_range.start = 0
    p.yaxis.axis_label = 'Frequency'
    p.toolbar.autohide = True

    if x_ticks is not None:
        p.xaxis.ticker = x_ticks

    p.xaxis.major_label_orientation = math.pi / 4
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    _ = p.multi_line(xs='xs', ys='ys', legend_field='label', line_color='color', source=data_source)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.0

    container.figure = p
    container.handle = bokeh.plotting.show(p, notebook_handle=True)
    container.data_source = data_source


def plot(data, **kwargs):
    container = kwargs['container']
    container.data_source.data.update(data)
    bokeh.io.push_notebook(handle=container.handle)
