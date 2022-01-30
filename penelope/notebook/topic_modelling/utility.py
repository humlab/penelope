from __future__ import annotations

import bqplot as bq
import ipydatagrid as dg
import pandas as pd


def table_widget(data: pd.DataFrame, **kwargs) -> None:  # pylint: disable=unused-argument
    weight_renderer: dg.TextRenderer = dg.TextRenderer(
        text_color="blue", background_color=bq.ColorScale(min=-3.0, max=1.0)
    )
    renderers = {
        "weight": weight_renderer,
    }
    grid: dg.DataGrid = dg.DataGrid(data, renderers=renderers, selection_mode="cell")

    grid.auto_fit_params = {"area": "body"}
    grid.auto_fit_columns = True
    grid.editable = False
    return grid
