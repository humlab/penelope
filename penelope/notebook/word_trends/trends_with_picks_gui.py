import abc
from typing import Any, Callable, List, Sequence

import pandas as pd
from bokeh.plotting import show
from ipywidgets import HTML, Button, HBox, IntSlider, Layout, Output, SelectMultiple, Text, ToggleButton, VBox

from penelope.corpus import VectorizedCorpus

from .displayers import plotter
from .interface import TrendsData


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
    def get_tokens_slice(self, start: int = 0, n: int = 0):
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


class SelectMultipleTokensSelector(TokensSelector):
    def __init__(self, tokens: pd.DataFrame, token_column='l2_norm_token', norms_columns=None):
        self._text_widget: Text = None
        self._tokens_widget: SelectMultiple = None
        self._token_to_index = None
        super().__init__(tokens, token_column, norms_columns)

    def display(self, tokens: pd.DataFrame) -> "TokensSelector":
        super().display(tokens)
        if self.tokens is not None:
            self._rehash_token_to_index()
        return self

    def _rehash_token_to_index(self):
        self._token_to_index = {w: i for i, w in enumerate(self.tokens.index.tolist())}

    def _create_widget(self) -> SelectMultiple:

        _tokens = list(self.tokens.index)
        _layout = Layout(width="200px")
        self._tokens_widget = SelectMultiple(options=_tokens, value=[], rows=30)
        self._tokens_widget.layout = _layout
        self._tokens_widget.observe(self._on_selection_changed, "value")

        self._text_widget = Text(description="")
        self._text_widget.layout = _layout
        self._text_widget.observe(self._on_filter_changed, "value")

        _widget = VBox([HTML("<b>Filter</b>"), self._text_widget, self._tokens_widget])

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

    def get_tokens_slice(self, start: int = 0, n: int = 0):

        return self._tokens_widget.options[start : start + n]

    def set_selected_indices(self, indices: List[int]):
        _options = self._tokens_widget.options
        self._tokens_widget.value = [_options[i] for i in indices]

    def __len__(self) -> int:
        return len(self._tokens_widget.options)

    def __getitem__(self, key) -> str:
        return self._tokens_widget.options[key]


class TrendsWithPickTokensGUI:
    def __init__(self, token_selector: TokensSelector, update_handler: Callable = None):
        self.token_selector: TokensSelector = token_selector
        self.update_handler: Callable = update_handler

        self._page_size = IntSlider(
            description="Count",
            min=0,
            max=100,
            step=1,
            value=3,
            continuous_update=False,
            layout=Layout(width="300px"),
        )
        self._forward = Button(
            description=">>",
            button_style="Success",
            layout=Layout(width="40px", color="green"),
        )
        self._back = Button(
            description="<<",
            button_style="Success",
            layout=Layout(width="40px", color="green"),
        )
        self._split = ToggleButton(description="Split", layout=Layout(width="80px", color="green"))
        self._output = Output(layout=Layout(width="80%"))

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
        return VBox(
            [
                HBox([self._back, self._forward, self._page_size, self._split]),
                HBox([self.token_selector.widget, self._output], layout=Layout(width="98%")),
            ]
        )

    def display(self, trends_data: TrendsData):
        self.token_selector.display(trends_data.gof_data.most_deviating_overview)

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
