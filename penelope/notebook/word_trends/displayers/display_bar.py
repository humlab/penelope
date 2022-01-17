import math
from typing import Iterable, List

import bokeh.models as bm
import bokeh.plotting as bp
from bokeh.io import show

from ...utility import generate_colors
from .compile_mixins import UnstackedTabularCompileMixIn
from .interface import ITrendDisplayer
from .utils import generate_temporal_ticks


class BarDisplayer(UnstackedTabularCompileMixIn, ITrendDisplayer):
    def __init__(self, name: str = "Bar", **opts):
        super().__init__(name=name, **opts)
        self.year_tick: int = 5

    def setup(self):
        return

    def plot(self, *, plot_data: dict, temporal_key: str, **_):

        value_fields: List[str] = [w for w in plot_data.keys() if w not in (temporal_key, 'year')]
        colors: Iterable[str] = generate_colors(len(value_fields))
        source = bm.ColumnDataSource(data=plot_data)

        p: bp.Figure = bp.figure(plot_height=self.height, plot_width=self.width, title="TF")

        # offset: float = -0.25
        v: List[bm.GlyphRenderer] = []
        for value_field in value_fields:
            w: bm.GlyphRenderer = p.vbar(x=temporal_key, top=value_field, width=0.2, source=source, color=next(colors))
            # offset += 0.25
            v.append(w)

        p.x_range.range_padding = 0.04
        p.xaxis.major_label_orientation = math.pi / 4
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.xaxis.ticker = generate_temporal_ticks(plot_data[temporal_key])

        legend: bm.Legend = bm.Legend(items=[(x, [v[i]]) for i, x in enumerate(value_fields)])
        p.add_layout(legend, 'center')
        p.legend.location = "top_left"

        with self.output:
            show(p)
