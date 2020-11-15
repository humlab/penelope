import itertools
import math
from dataclasses import dataclass, field

import bokeh

from ._displayer import ITrendDisplayer, YearTokenDataMixin


@dataclass
class BarDisplayer(YearTokenDataMixin, ITrendDisplayer):

    year_tick: int = field(default=5)
    name: str = field(default="Bar")

    def setup(self):
        return

    def plot(self, data, **_):

        tokens = [w for w in data.keys() if w != 'year']

        source = bokeh.models.ColumnDataSource(data=data)

        p = bokeh.plotting.figure(plot_height=400, plot_width=1000, title="Word frequecy by year")

        colors = itertools.islice(itertools.cycle(bokeh.palettes.d3['Category20'][20]), len(tokens))

        offset = -0.25
        v = []
        for token in tokens:
            w = p.vbar(x='year', top=token, width=0.2, source=source, color=next(colors))
            offset += 0.25
            v.append(w)

        p.x_range.range_padding = 0.04
        p.xaxis.major_label_orientation = math.pi / 4
        p.xaxis.ticker = self._year_ticks()

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        legend = bokeh.models.Legend(items=[(x, [v[i]]) for i, x in enumerate(tokens)])

        p.add_layout(legend, 'center')

        p.legend.location = "top_left"

        with self.output:
            bokeh.io.show(p)

    def _year_ticks(self):
        year_min, year_max = self.data.corpus.year_range()
        y_min = year_min - (year_min % self.year_tick)
        y_max = year_max if year_max % self.year_tick == 0 else year_max + (self.year_tick - year_max % self.year_tick)
        return list(range(y_min, y_max + 1, self.year_tick))


# @dataclass
# class BarDisplayer(YearTokenDataMixin, ITrendDisplayer):

#     year_tick: int = field(default=5)
#     name: str = field(default="Bar")

#     # FIXME: Implement https://morioh.com/p/9252315eb8d6 instead?
#     def setup(self):
#         return

#     def plot(self, data, **_):

#         years = [str(y) for y in data['year']]

#         data['year'] = years

#         tokens = [w for w in data.keys() if w != 'year']

#         source = bokeh.models.ColumnDataSource(data=data)

#         max_value = max([max(data[key]) for key in data if key != 'year']) + 0.005

#         p = bokeh.plotting.figure(
#             x_range=years, y_range=(0, max_value), plot_height=400, plot_width=1200, title="Word frequecy by year"
#         )

#         colors = itertools.islice(itertools.cycle(bokeh.palettes.d3['Category20'][20]), len(tokens))

#         offset = -0.25
#         v = []
#         for token in tokens:
#             w = p.vbar(
#                 x=bokeh.transform.dodge('year', offset, range=p.x_range),
#                 top=token,
#                 width=0.2,
#                 source=source,
#                 color=next(colors),
#             )
#             offset += 0.25
#             v.append(w)

#         p.x_range.range_padding = 0.04
#         p.xaxis.major_label_orientation = math.pi / 4
#         # p.xaxis.ticker = self._year_ticks()
#         # p.xgrid.ticker = p.xaxis[0].ticker
#         # p.xgrid.ticker = MyTicker()

#         p.xgrid.grid_line_color = None
#         p.ygrid.grid_line_color = None

#         legend = bokeh.models.Legend(items=[(x, [v[i]]) for i, x in enumerate(tokens)])

#         p.add_layout(legend, 'center')
#         p.legend.location = "top_left"

#         # p.add_layout(legend, 'left')
#         # p.legend.location = "top_right"
#         # p.legend.orientation = "vertical"

#         with self.output:
#             bokeh.io.show(p)

#     def _year_ticks(self):
#         year_min, year_max = self.data.corpus.year_range()
#         y_min = year_min - (year_min % self.year_tick)
#         y_max = year_max if year_max % self.year_tick == 0 else year_max + (self.year_tick - year_max % self.year_tick)
#         return list(map(str, range(y_min, y_max + 1, self.year_tick)))
