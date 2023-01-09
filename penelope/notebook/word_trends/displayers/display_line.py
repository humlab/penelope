import math
from typing import Any, List, Sequence

import bokeh.models as bm
import bokeh.plotting as bp
import ipywidgets
import pandas as pd
from bokeh.io import push_notebook

from penelope.plot.utility import generate_colors, generate_temporal_ticks

from .interface import ITrendDisplayer


class LineDisplayer(ITrendDisplayer):
    def __init__(self, name: str = "Line", **opts):
        super().__init__(name=name, **opts)

        self.chart: bp.figure = None
        self.handle: Any = None
        self.data_source: bm.ColumnDataSource = None
        self.year_tick: int = 5

    def setup(self):

        self.output = ipywidgets.Output()
        self.data_source = bm.ColumnDataSource(dict(xs=[[0]], ys=[[0]], labels=[""], colors=['red']))
        self.chart = self.figure(data_source=self.data_source)

    def figure(self, *, data_source: bm.ColumnDataSource, x_ticks: Sequence[int] = None):

        p = bp.figure(width=self.width, height=self.height, sizing_mode='scale_width')

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

    def plot(self, *, data: Sequence[pd.DataFrame], temporal_key: str, **_) -> None:
        """Data is unstacked i.e. columns are token@pivot_keys"""
        unstacked_data: pd.DataFrame = data[-1]

        if temporal_key in unstacked_data.columns:
            unstacked_data = unstacked_data.set_index(temporal_key, drop=True)

        data: dict = self._compile(unstacked_data=unstacked_data)

        if self.handle is None:
            self.handle = bp.show(self.chart, notebook_handle=True)

        if len(data['xs']) > 0:
            self.chart.xaxis.ticker = generate_temporal_ticks(data['xs'][0])

        self.data_source.data.update(data)

        push_notebook(handle=self.handle)

        # ipydisplay(unstacked_data)
        # ipydisplay(data)

    def clear(self):
        return

    def _compile(self, *, unstacked_data: pd.DataFrame) -> dict:
        """Compile multiline plot data for token ids `indices`, apply `smoothers` functions. Return dict"""
        temporal_categories = unstacked_data.index.tolist()

        xs_data, ys_data = [], []
        for token_column in unstacked_data.columns:

            # smoothers: List[Callable] = kwargs.get('smoothers', []) or []

            xs_j, ys_j = temporal_categories, unstacked_data[token_column]

            # for smoother in smoothers:
            #     xs_j, ys_j = smoother(xs_j, ys_j)

            xs_data.append(xs_j)
            ys_data.append(ys_j)

        labels: List[str] = unstacked_data.columns.tolist()
        colors: List[str] = generate_colors(n=len(unstacked_data.columns), palette_id=20)

        data: dict = dict(xs=xs_data, ys=ys_data, labels=labels, colors=colors)

        return data
