import abc
import types
from typing import Any, List, Sequence

import ipywidgets
import pandas as pd
import penelope.plot.word_trend_plot as plotter
import qgrid
from bokeh.plotting import show
from IPython.display import display
from penelope.corpus import VectorizedCorpus


class TokensSelector(abc.ABC):
    def __init__(self, tokens: pd.DataFrame, token_column='l2_norm_token', norms_columns=None):

        self.tokens = tokens
        self.token_column = token_column
        if self.token_column in self.tokens.columns:
            self.tokens = self.tokens.set_index(self.token_column)
        _gof_value_columns = norms_columns or (
            'l2_norm',
            'abs_l2_norm',
        )
        self.tokens = self.tokens.rename(columns={_gof_value_columns[0]: "GoF", _gof_value_columns[1]: "|GoF|"})
        self._widget = None
        self._on_selection_change_handler = None

    @property
    def widget(self):
        if self._widget is None:
            self._widget = self._create_widget()
        return self._widget

    def on_selection_change_handler(self, handler):
        self._on_selection_change_handler = handler

    def _on_selection_changed(self, *_args):
        if self._on_selection_change_handler is not None:
            self._on_selection_change_handler()

    @abc.abstractmethod
    def _create_widget(self) -> Any:
        return None

    @abc.abstractmethod
    def get_selected_tokens(self) -> List[str]:
        return None

    @abc.abstractmethod
    def get_selected_indices(self) -> List[int]:
        return None

    @abc.abstractmethod
    def get_tokens_slice(self, start: int = 0, n_count: int = 0):
        return None

    @abc.abstractmethod
    def set_selected_indices(self, indices: List[int]):
        return

    @abc.abstractmethod
    def __len__(self) -> int:
        return 0

    @abc.abstractmethod
    def __getitem__(self, key) -> str:
        return None


class QgridTokensSelector(TokensSelector):
    def __init__(self, tokens: pd.DataFrame, token_column='l2_norm_token', norms_columns=None):

        super().__init__(tokens, token_column, norms_columns)
        self._widget: qgrid.QGridWidget = None

    @property
    def widget(self):
        if self._widget is None:
            self._widget = self._create_widget()
        return self._widget

    def on_selection_change_handler(self, handler):
        self._on_selection_change_handler = handler

    def _on_selection_changed(self, *_args):
        if self._on_selection_change_handler is not None:
            self._on_selection_change_handler(_args[0]['new'])

    def _create_widget(self) -> qgrid.QGridWidget:

        grid_options = {
            # SlickGrid options
            # https://github.com/6pac/SlickGrid/wiki/Grid-Options
            'fullWidthRows': False,
            'syncColumnCellResize': True,
            'forceFitColumns': False,
            'defaultColumnWidth': 150,
            'rowHeight': 28,
            'enableColumnReorder': False,
            'enableTextSelectionOnCells': True,
            # 'editable': True,
            'autoEdit': False,
            # 'explicitInitialization': True,
            # Qgrid options
            'maxVisibleRows': 30,
            'minVisibleRows': 20,
            'sortable': True,
            'filterable': True,
            'highlightSelectedCell': False,
            'highlightSelectedRow': True,
        }

        # qgrid.enable(dataframe=True, series=True)

        col_opts = {
            # https://github.com/6pac/SlickGrid/wiki/Column-Options
            'editable': False
        }

        column_definitions = {
            # https://github.com/6pac/SlickGrid/wiki/Column-Options
            self.token_column: {
                'defaultSortAsc': True,
                'maxWidth': 300,
                'minWidth': 180,
                'resizable': True,
                'sortable': True,
                # 'toolTip': "",
                'width': 180,
                # 'editable': True
            },
            'GoF': {
                # 'defaultSortAsc': True,
                'maxWidth': 80,
                'minWidth': 20,
                'resizable': True,
                'sortable': True,
                # 'toolTip': "",
                'width': 70,
                'editable': False,
                # 'filterable': False,
            },
            '|GoF|': {
                # 'defaultSortAsc': True,
                'maxWidth': 80,
                'minWidth': 20,
                'resizable': True,
                'sortable': True,
                # 'toolTip': "",
                'width': 70,
                'editable': False,
                # 'filterable': False,
            },
        }

        q = qgrid.show_grid(
            self.tokens,
            precision=6,
            grid_options=grid_options,
            column_options=col_opts,
            column_definitions=column_definitions,
        )

        q.layout = ipywidgets.Layout(width="450px")

        q.on('selection_changed', self._on_selection_changed)

        return q

    def get_selected_tokens(self) -> List[str]:

        return list(self._widget.get_selected_df().index)

    def get_selected_indices(self) -> List[int]:

        return self._widget.get_selected_rows()

    def get_tokens_slice(self, start: int = 0, n_count: int = 0):
        return list(self.tokens.iloc[start : start + n_count].index)

    def set_selected_indices(self, indices: List[int]):
        return self._widget.change_selection(self.tokens.iloc[indices].index)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, key):
        return self.tokens.iloc[key].index


class SelectMultipleTokensSelector(TokensSelector):
    def __init__(self, tokens: pd.DataFrame, token_column='l2_norm_token', norms_columns=None):
        super().__init__(tokens, token_column, norms_columns)
        self._text_widget: ipywidgets.Text = None
        self._tokens_widget: ipywidgets.SelectMultiple = None
        self._token_to_index = self._rehash_token_to_index()

    def _rehash_token_to_index(self):
        self._token_to_index = {w: i for i, w in enumerate(self.tokens.index.tolist())}
        return self._token_to_index

    def _create_widget(self) -> ipywidgets.SelectMultiple:

        _tokens = list(self.tokens.index)

        self._tokens_widget = ipywidgets.SelectMultiple(options=_tokens, value=[], rows=30)
        self._tokens_widget.layout = ipywidgets.Layout(width="250px")
        self._tokens_widget.observe(self._on_selection_changed, "value")

        self._text_widget = ipywidgets.Text(description="Filter:")
        self._text_widget.layout = ipywidgets.Layout(width="250px")
        self._text_widget.observe(self._on_filter_changed, "value")

        _widget = ipywidgets.VBox([self._text_widget, self._tokens_widget])

        return _widget

    def _on_filter_changed(self, *_):
        _filter = self._text_widget.value.strip()
        if _filter == "":
            _options = self.tokens.index.tolist()
        else:
            _options = self.tokens[self.tokens.index.str.contains(_filter)].index.tolist()
        self._tokens_widget.value = [x for x in self._tokens_widget.value if x in _options]
        self._tokens_widget.options = _options

    def get_selected_tokens(self) -> List[str]:

        return list(self._tokens_widget.value)

    def get_selected_indices(self) -> List[int]:

        return [self._token_to_index[w] for w in self._tokens_widget.value]

    def get_tokens_slice(self, start: int = 0, n_count: int = 0):

        return self._tokens_widget.options[start : start + n_count]

    def set_selected_indices(self, indices: List[int]):
        _options = self._tokens_widget.options
        self._tokens_widget.value = [_options[i] for i in indices]

    def __len__(self) -> int:
        return len(self._tokens_widget.options)

    def __getitem__(self, key) -> str:
        return self._tokens_widget.options[key]


def display_gui(
    x_corpus: VectorizedCorpus,
    df_tokens: pd.DataFrame,
    n_columns: int = 3,
    token_sector_cls: TokensSelector = SelectMultipleTokensSelector,
):

    gui = types.SimpleNamespace(
        progress=ipywidgets.IntProgress(
            description="",
            min=0,
            max=10,
            step=1,
            value=0,
            continuous_update=False,
            layout=ipywidgets.Layout(width="98%"),
        ),
        n_count=ipywidgets.IntSlider(
            description="Count",
            min=0,
            max=100,
            step=1,
            value=3,
            continuous_update=False,
            layout=ipywidgets.Layout(width="300px"),
        ),
        forward=ipywidgets.Button(
            description=">>",
            button_style="Success",
            layout=ipywidgets.Layout(width="40px", color="green"),
        ),
        back=ipywidgets.Button(
            description="<<",
            button_style="Success",
            layout=ipywidgets.Layout(width="40px", color="green"),
        ),
        split=ipywidgets.ToggleButton(description="Split", layout=ipywidgets.Layout(width="80px", color="green")),
        output=ipywidgets.Output(layout=ipywidgets.Layout(width="99%")),
    )

    token_selector = token_sector_cls(df_tokens)

    def update_plot(*_):

        gui.output.clear_output()

        selected_tokens: Sequence[str] = token_selector.get_selected_tokens()

        if len(selected_tokens) == 0:
            selected_tokens = token_selector[: gui.n_count.value]

        indices: List[int] = [x_corpus.token2id[token] for token in selected_tokens]

        with gui.output:
            x_columns: int = n_columns if gui.split.value else None
            p = plotter.yearly_token_distribution_multiple_line_plot(
                x_corpus, indices, width=1000, height=600, n_columns=x_columns
            )
            show(p)

    def stepper_clicked(b):

        _selected_indices = token_selector.get_selected_indices()
        _current_index = min(_selected_indices) if len(_selected_indices) > 0 else 0

        if b.description == "<<":
            _current_index = max(_current_index - gui.n_count.value, 0)

        if b.description == ">>":
            _current_index = min(_current_index + gui.n_count.value, len(token_selector) - gui.n_count.value)

        token_selector.set_selected_indices(list(range(_current_index, _current_index + gui.n_count.value)))

    def split_changed(*_):
        update_plot()

    gui.n_count.observe(update_plot, "value")
    gui.split.observe(split_changed, "value")
    gui.forward.on_click(stepper_clicked)
    gui.back.on_click(stepper_clicked)

    token_selector.on_selection_change_handler(update_plot)

    display(
        ipywidgets.VBox(
            [
                gui.progress,
                ipywidgets.HBox([gui.back, gui.forward, gui.n_count, gui.split]),
                ipywidgets.HBox([token_selector.widget, gui.output]),
            ]
        )
    )
    update_plot()
