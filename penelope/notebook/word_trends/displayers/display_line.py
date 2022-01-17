import math
from typing import Any, Sequence, Union

import bokeh.models as bm
import bokeh.plotting as bp
import ipywidgets
import pandas as pd
from bokeh.io import push_notebook

from .compile_mixins import MultiLineCompileMixIn
from .interface import ITrendDisplayer
from .utils import generate_temporal_ticks


class LineDisplayer(MultiLineCompileMixIn, ITrendDisplayer):
    def __init__(self, name: str = "Line", **opts):
        super().__init__(name=name, **opts)

        self.chart: bp.Figure = None
        self.handle: Any = None
        self.data_source: bm.ColumnDataSource = None
        self.year_tick: int = 5

    def setup(self):

        self.output = ipywidgets.Output()
        self.data_source = bm.ColumnDataSource(dict(xs=[[0]], ys=[[0]], labels=[""], colors=['red']))
        self.chart = self.figure(data_source=self.data_source)

    def figure(self, *, data_source: bm.ColumnDataSource, x_ticks: Sequence[int] = None):

        p = bp.figure(plot_width=self.width, plot_height=self.height)

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

        _ = p.multi_line(xs='xs', ys='ys', legend_field='labels', line_color='colors', source=data_source)

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.background_fill_alpha = 0.0

        return p

    def plot(self, *, plot_data: Union[pd.DataFrame, dict], **_):

        data: dict = plot_data if isinstance(plot_data, dict) else {x: plot_data[x] for x in plot_data.columns}

        if self.handle is None:
            self.handle = bp.show(self.chart, notebook_handle=True)

        if len(data['xs']) > 0:
            self.chart.xaxis.ticker = generate_temporal_ticks(data['xs'][0])

        self.data_source.data.update(data)

        push_notebook(handle=self.handle)

    def clear(self):
        return
