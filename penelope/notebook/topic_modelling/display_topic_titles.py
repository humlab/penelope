from typing import Callable

import pandas as pd
from ipyaggrid import Grid
from IPython.core.display import Javascript
from IPython.display import display as IPython_display
from ipywidgets import HBox, IntSlider, Output, Text, VBox
from ipywidgets.widgets.widget_button import Button
from penelope.notebook.utility import create_js_download

from .utility import reduce_topic_tokens_overview

# FIXME #89 Replace ipyaggrid with `perspective` or `panel.Tabulator`
CSS_RULES = ".ag-cell { white-space: normal !important; } "
DEFAULT_STYLE = dict(
    export_csv=True,
    export_excel=True,
    export_mode="buttons",
    index=True,
    keep_multiindex=False,
    menu={"buttons": [{"name": "Export Grid", "hide": True}]},
    quick_filter=True,
    show_toggle_delete=False,
    show_toggle_edit=False,
    theme="ag-theme-balham",
)
GRID_OPTIONS = dict(
    columnDefs=[
        {"headerName": "Topic", "field": "topic_id", "maxWidth": 80},
        {"headerName": "Tokens", "field": "tokens"},
    ],
    enableSorting=True,
    enableFilter=True,
    enableColResize=True,
    enableRangeSelection=False,
    rowSelection='multiple',
    defaultColDef={
        "flex": 1,
        "wrapText": True,
        "autoHeight": True,
        "sortable": True,
        "resizable": True,
        "editable": False,
    },
    onColumnResized="function onColumnResized(params) {params.api.resetRowHeights();params.api.setDomLayout('autoHeight');}",
    onColumnVisible="function onColumnResized(params) {params.api.resetRowHeights();params.api.setDomLayout('autoHeight');}",
)


class DisplayIPyWidgetsGUI:
    def __init__(self):
        self.count_slider: IntSlider = IntSlider(description="Tokens", min=25, max=200, value=50)
        self.output = Output()
        self.data = None

    def layout(self):
        return VBox([self.count_slider, self.output])

    def display_table(self, df):
        g = Grid(
            grid_data=df,
            grid_options=GRID_OPTIONS,
            **DEFAULT_STYLE,
            css_rules=CSS_RULES,
        )
        IPython_display(g)


PANDAS_TABLE_STYLE = [
    dict(
        selector="th",
        props=[
            ('font-size', '11px'),
            ('text-align', 'left'),
            ('font-weight', 'bold'),
            ('color', '#6d6d6d'),
            ('background-color', '#f7f7f9'),
        ],
    ),
    dict(
        selector="td",
        props=[
            ('font-size', '11px'),
            ('text-align', 'left'),
        ],
    ),
]


class DisplayPandasGUI:  # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self.count_slider: IntSlider = IntSlider(
            description="Tokens",
            min=25,
            max=200,
            value=50,
            continuous_update=False,
        )
        self.search_text: Text = Text(description="Find")
        self.download_button: Button = Button(description="Download")
        self.output: Output = Output()
        self.topics: pd.DataFrame = None
        self.reduced_topics: pd.DataFrame = None
        self.callback: Callable = lambda *_: ()
        self.js_download: Javascript = None

    def layout(self):
        return VBox((HBox((self.count_slider, self.search_text, self.download_button)), self.output))

    def download(self, *_):
        with self.output:
            js_download = create_js_download(self.reduced_topics, index=True)
            if js_download is not None:
                IPython_display(js_download)

    def update(self, *_):

        self.output.clear_output()

        with self.output:

            self.reduced_topics = reduce_topic_tokens_overview(
                self.topics, self.count_slider.value, self.search_text.value
            )

            styled_reduced_topics = self.reduced_topics.style.set_table_styles(PANDAS_TABLE_STYLE)

            IPython_display(styled_reduced_topics)

    def display(self, topics: pd.DataFrame, callback: Callable = None):

        self.callback = callback or self.callback
        self.topics = topics
        self.count_slider.observe(self.update, "value")
        self.search_text.observe(self.update, "value")
        self.download_button.on_click(self.download)

        # pd.options.display.max_colwidth = None
        pd.set_option('colheader_justify', 'left')

        IPython_display(self.layout())

        return self


def display_gui(topics: pd.DataFrame, displayer_cls: type):

    _ = displayer_cls().display(topics=topics, callback=reduce_topic_tokens_overview)
