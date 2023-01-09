import math
from typing import Iterable, List, Sequence

import bokeh.models as bm
import bokeh.plotting as bp
import pandas as pd
from bokeh.io import show

from penelope.plot.utility import generate_colors, generate_temporal_ticks

from .interface import ITrendDisplayer


class BarDisplayer(ITrendDisplayer):
    def __init__(self, name: str = "Bar", **opts):
        super().__init__(name=name, **opts)
        self.year_tick: int = 5

    def setup(self):
        return

    # def compile(
    #     self,
    #     unstacked_data: pd.DataFrame,
    #     temporal_key: str,
    #     **_,
    # ) -> pd.DataFrame:
    #     """Extracts trend vectors for tokens ´indices` and returns a pd.DataFrame."""

    #     data = {
    #         **{temporal_key: unstacked_data.index},
    #         **{unstacked_data[col] for col in unstacked_data.columns if col},
    #     }

    #     return pd.DataFrame(data=data)

    def plot(self, *, data: Sequence[pd.DataFrame], temporal_key: str, **_) -> None:

        unstacked_data: pd.DataFrame = data[-1]

        # ipydisplay(unstacked_data)

        if temporal_key in unstacked_data.columns:
            unstacked_data.set_index(temporal_key, drop=True)

        plot_data: dict = {
            **{column: unstacked_data[column] for column in unstacked_data.columns},
            **{temporal_key: [str(x) for x in unstacked_data.index]},
        }

        value_fields: List[str] = [w for w in plot_data.keys() if w not in (temporal_key, 'year')]
        colors: Iterable[str] = iter(generate_colors(len(value_fields)))
        source = bm.ColumnDataSource(data=plot_data)

        p: bp.figure = bp.figure(height=self.height, width=self.width, sizing_mode='scale_width', title="TF")

        offset: float = -0.25
        v: List[bm.GlyphRenderer] = []
        for value_field in value_fields:
            w: bm.GlyphRenderer = p.vbar(x=temporal_key, top=value_field, width=1.0, source=source, color=next(colors))
            offset += 0.25
            v.append(w)

        p.x_range.range_padding = 0.00
        p.xaxis.major_label_orientation = math.pi / 4
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.xaxis.ticker = generate_temporal_ticks(plot_data[temporal_key])

        legend: bm.Legend = bm.Legend(items=[(x, [v[i]]) for i, x in enumerate(value_fields)])
        p.add_layout(legend, 'center')
        p.legend.location = "top_left"

        with self.output:
            show(p)
