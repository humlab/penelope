import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import bokeh
import bokeh.models
import bokeh.plotting
import ipywidgets

from ._compile_mixins import LinesDataMixin
from ._displayer import ITrendDisplayer
from .utils import get_year_category_ticks

PLOT_WIDTH = 800
PLOT_HEIGHT = 500


@dataclass
class LineDisplayer(LinesDataMixin, ITrendDisplayer):

    name: str = field(default="Line")
    figure: bokeh.plotting.Figure = None
    handle: Any = None
    data_source: bokeh.models.ColumnDataSource = None
    year_tick: int = field(default=5)

    def setup(self):

        self.output = ipywidgets.Output()
        self.data_source = bokeh.models.ColumnDataSource({'xs': [[0]], 'ys': [[0]], 'label': [""], 'color': ['red']})
        self.figure = self._setup_plot(data_source=self.data_source)

    def _setup_plot(
        self,
        *,
        data_source: bokeh.models.ColumnDataSource,
        plot_width=PLOT_WIDTH,
        plot_height=PLOT_HEIGHT,
        x_ticks: Sequence[int] = None,
    ):

        p = bokeh.plotting.figure(plot_width=plot_width, plot_height=plot_height)

        p.y_range.start = 0
        p.yaxis.axis_label = 'Frequency'
        p.toolbar.autohide = True

        if x_ticks is not None:
            p.xaxis.ticker = x_ticks
            # p.xaxis.ticker = list(range(1920,2021, 5))
            # p.xaxis.ticker = FixedTicker(ticks=list(range(1920,2021, 5)))
            # p.xaxis.major_label_overrides = {x: str(x) if x % 5==0 else '' for x in xs }

        p.xaxis.major_label_orientation = math.pi / 4
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        # Vertical grid lines at each tick
        # p.xgrid.grid_line_color = "black"
        # p.xgrid.grid_line_alpha = 0.5
        # p.xgrid.grid_line_dash = [6, 4]

        # alternating bands between ticks
        p.xgrid.band_fill_alpha = 0.01
        p.xgrid.band_fill_color = "black"

        _ = p.multi_line(xs='xs', ys='ys', legend_field='label', line_color='color', source=data_source)

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.background_fill_alpha = 0.0

        return p

    def plot(self, plot_data: dict, **_):  # pylint: disable=unused-argument

        if self.handle is None:
            self.handle = bokeh.plotting.show(self.figure, notebook_handle=True)

        if len(plot_data['xs']) > 0:
            self.figure.xaxis.ticker = get_year_category_ticks(plot_data['xs'][0])

        self.data_source.data.update(plot_data)
        bokeh.io.push_notebook(handle=self.handle)

    def clear(self):
        return
