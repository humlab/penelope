import abc
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import ipywidgets
import pandas as pd
import qgrid
from bokeh.plotting import show
from penelope.corpus import VectorizedCorpus
from penelope.notebook.word_trends.trends_data import TrendsData

from .displayers import deprecated_plot as plotter


class TokensSelector(abc.ABC):
    def __init__(self, tokens: pd.DataFrame, token_column='l2_norm_token', norms_columns=None):

        self.tokens = tokens
        self.token_column = token_column
        self.norms_columns = norms_columns or (
            'l2_norm',
            'abs_l2_norm',
        )
        self._widget = None
        self._on_selection_change_handler = None

        self.display(tokens)

    def display(self, tokens: pd.DataFrame) -> "TokensSelector":

        if tokens is None:
            return self

        self.tokens = tokens
        if self.token_column in self.tokens.columns:
            self.tokens = self.tokens.set_index(self.token_column)
        self.tokens = self.tokens.rename(columns={self.norms_columns[0]: "GoF", self.norms_columns[1]: "|GoF|"})
        self._widget = None
        return self

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
            'maxVisibleRows': 20,
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

        q.layout = ipywidgets.Layout(width="250px")

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
        self._text_widget: ipywidgets.Text = None
        self._tokens_widget: ipywidgets.SelectMultiple = None
        self._token_to_index = None
        super().__init__(tokens, token_column, norms_columns)

    def display(self, tokens: pd.DataFrame) -> "TokensSelector":
        super().display(tokens)
        if self.tokens is not None:
            self._rehash_token_to_index()
        return self

    def _rehash_token_to_index(self):
        self._token_to_index = {w: i for i, w in enumerate(self.tokens.index.tolist())}

    def _create_widget(self) -> ipywidgets.SelectMultiple:

        _tokens = list(self.tokens.index)
        _layout = ipywidgets.Layout(width="200px")
        self._tokens_widget = ipywidgets.SelectMultiple(options=_tokens, value=[], rows=30)
        self._tokens_widget.layout = _layout
        self._tokens_widget.observe(self._on_selection_changed, "value")

        self._text_widget = ipywidgets.Text(description="")
        self._text_widget.layout = _layout
        self._text_widget.observe(self._on_filter_changed, "value")

        _widget = ipywidgets.VBox([ipywidgets.HTML("<b>Filter</b>"), self._text_widget, self._tokens_widget])

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


@dataclass
class TrendsWithPickTokensGUI:

    token_selector: TokensSelector = None
    update_handler: Callable = None

    _page_size = ipywidgets.IntSlider(
        description="Count",
        min=0,
        max=100,
        step=1,
        value=3,
        continuous_update=False,
        layout=ipywidgets.Layout(width="300px"),
    )
    _forward = ipywidgets.Button(
        description=">>",
        button_style="Success",
        layout=ipywidgets.Layout(width="40px", color="green"),
    )
    _back = ipywidgets.Button(
        description="<<",
        button_style="Success",
        layout=ipywidgets.Layout(width="40px", color="green"),
    )
    _split = ipywidgets.ToggleButton(description="Split", layout=ipywidgets.Layout(width="80px", color="green"))
    _output = ipywidgets.Output(layout=ipywidgets.Layout(width="80%"))

    def setup(self):

        self._page_size.observe(self._update, "value")
        self._split.observe(self.split_changed, "value")
        self._forward.on_click(self._stepper_clicked)
        self._back.on_click(self._stepper_clicked)
        self.token_selector.on_selection_change_handler(self._update)

    def _stepper_clicked(self, b):

        _selected_indices = self.token_selector.get_selected_indices()
        _current_index = min(_selected_indices) if len(_selected_indices) > 0 else 0

        if b.description == "<<":
            _current_index = max(_current_index - self.page_size, 0)

        if b.description == ">>":
            _current_index = min(_current_index + self.page_size, len(self.token_selector) - self.page_size)

        self.token_selector.set_selected_indices(list(range(_current_index, _current_index + self.page_size)))

    def _update(self):
        if self.update_handler is not None:
            with self._output:
                tokens = self.selected_tokens()
                self.update_handler(self, tokens)

    def split_changed(self, *_):
        self._update()

    def layout(self):
        return ipywidgets.VBox(
            [
                ipywidgets.HBox([self._back, self._forward, self._page_size, self._split]),
                ipywidgets.HBox([self.token_selector.widget, self._output], layout=ipywidgets.Layout(width="98%")),
            ]
        )

    def display(self, trends_data: TrendsData):
        self.token_selector.display(trends_data.most_deviating_overview)

    @property
    def page_size(self) -> int:
        return self._page_size.value

    @property
    def split(self) -> bool:
        return self._split.value

    @property
    def selected_tokens(self) -> Sequence[str]:
        tokens: Sequence[str] = self.token_selector.get_selected_tokens()
        if len(tokens) == 0:
            tokens = self.token_selector[: self.page_size]
        return tokens

    # FIXME: Make this a drop-in replacement for text entry i.e. no plotting of its own, just raise event with selected words
    @staticmethod
    def create(
        corpus: VectorizedCorpus,
        tokens: pd.DataFrame,
        n_columns: int = 3,
        token_sector_cls: TokensSelector = SelectMultipleTokensSelector,
        tokens_selected=None,
    ) -> "TrendsWithPickTokensGUI":

        gui = TrendsWithPickTokensGUI(
            token_selector=token_sector_cls(tokens),
            update_handler=lambda tokens, split: (tokens_selected or TrendsWithPickTokensGUI.default_tokens_plotter)(
                tokens=tokens, corpus=corpus, n_columns=n_columns, split=split
            ),
        )
        return gui

    @staticmethod
    def default_tokens_plotter(tokens: Sequence[str], corpus: VectorizedCorpus, n_columns: int, split: bool):
        indices: List[int] = [corpus.token2id[token] for token in tokens]
        # FIXME: Switch to the one used in TrendsGUI???
        p = plotter.yearly_token_distribution_multiple_line_plot(
            corpus,
            indices,
            width=1000,
            height=600,
            n_columns=n_columns if split else None,
        )
        if p is None:
            return
        show(p)
