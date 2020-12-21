import itertools
import math
from dataclasses import dataclass, field

import bokeh

from ._compile_mixins import CategoryDataMixin
from ._displayer import ITrendDisplayer
from .utils import get_year_category_ticks


@dataclass
class BarDisplayer(CategoryDataMixin, ITrendDisplayer):

    year_tick: int = field(default=5)
    name: str = field(default="Bar")

    def setup(self):
        return

    def plot(self, plot_data: dict, **_):  # pylint: disable=unused-argument

        tokens = [w for w in plot_data.keys() if w not in ('category', 'year')]

        source = bokeh.models.ColumnDataSource(data=plot_data)

        p = bokeh.plotting.figure(plot_height=400, plot_width=1000, title="Word frequecy")

        colors = itertools.islice(itertools.cycle(bokeh.palettes.d3['Category20'][20]), len(tokens))

        offset = -0.25
        v = []
        for token in tokens:
            w = p.vbar(x='category', top=token, width=0.2, source=source, color=next(colors))
            offset += 0.25
            v.append(w)

        p.x_range.range_padding = 0.04
        p.xaxis.major_label_orientation = math.pi / 4
        p.xaxis.ticker = get_year_category_ticks(plot_data['category'])

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        legend = bokeh.models.Legend(items=[(x, [v[i]]) for i, x in enumerate(tokens)])

        p.add_layout(legend, 'center')

        p.legend.location = "top_left"

        with self.output:
            bokeh.io.show(p)
