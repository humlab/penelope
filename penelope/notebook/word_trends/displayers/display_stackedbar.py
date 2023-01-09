import math
from typing import List, Sequence

import bokeh.models as bm
import bokeh.plotting as bp
import pandas as pd
from bokeh.io import show

from penelope.plot.utility import generate_colors

from .interface import ITrendDisplayer


class StackedBarDisplayer(ITrendDisplayer):
    def __init__(self, name: str = "Bar", **opts):
        super().__init__(name=name, **opts)
        self.year_tick: int = 5

    def setup(self):
        return

    def plot(self, *, data: Sequence[pd.DataFrame], temporal_key: str, **_) -> None:

        unstacked_data: pd.DataFrame = data[-1]

        if temporal_key in unstacked_data.columns:
            unstacked_data.set_index(temporal_key, drop=True)

        plot_data: dict = {
            **{column: unstacked_data[column] for column in unstacked_data.columns},
            **{temporal_key: [str(x) for x in unstacked_data.index]},
        }

        source = bm.ColumnDataSource(data=plot_data)

        temporal_values: List[str] = [str(x) for x in plot_data[temporal_key]]
        tokens: List[str] = [t for t in plot_data.keys() if t != temporal_key]
        colors: List[str] = generate_colors(len(tokens))

        p: bp.figure = bp.figure(
            x_range=temporal_values, height=self.height, width=self.width, sizing_mode='scale_width', title="TF"
        )

        p.vbar_stack(tokens, x=temporal_key, width=0.9, color=colors, source=source, legend_label=tokens)

        p.y_range.start = 0
        p.x_range.range_padding = 0.02
        p.xaxis.major_label_orientation = math.pi / 4
        p.xgrid.grid_line_color = None
        # p.ygrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        p.legend.location = "top_right"
        p.legend.orientation = "vertical"
        p.legend.background_fill_alpha = 0.0

        # p.xaxis.ticker = generate_temporal_ticks(plot_data[temporal_key])

        with self.output:
            show(p)
