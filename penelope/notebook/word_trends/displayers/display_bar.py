import itertools
import math

import bokeh

from ._compile_mixins import CategoryDataMixin
from .interface import ITrendDisplayer
from .utils import get_year_category_ticks


class BarDisplayer(CategoryDataMixin, ITrendDisplayer):
    def __init__(self, name: str = "Bar"):
        super().__init__(name=name)
        self.year_tick: int = 5

    def setup(self):
        return

    def plot(self, *, plot_data: dict, category_name: str, **_):

        tokens = [w for w in plot_data.keys() if w not in (category_name, 'year')]

        source = bokeh.models.ColumnDataSource(data=plot_data)

        p = bokeh.plotting.figure(plot_height=400, plot_width=1000, title="Word frequecy")

        colors = itertools.islice(itertools.cycle(bokeh.palettes.d3['Category20'][20]), len(tokens))

        offset = -0.25
        v = []
        for token in tokens:
            w = p.vbar(x=category_name, top=token, width=0.2, source=source, color=next(colors))
            offset += 0.25
            v.append(w)

        p.x_range.range_padding = 0.04
        p.xaxis.major_label_orientation = math.pi / 4
        p.xaxis.ticker = get_year_category_ticks(plot_data[category_name])

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        legend = bokeh.models.Legend(items=[(x, [v[i]]) for i, x in enumerate(tokens)])

        p.add_layout(legend, 'center')

        p.legend.location = "top_left"

        with self.output:
            bokeh.io.show(p)
