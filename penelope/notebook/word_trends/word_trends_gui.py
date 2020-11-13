from dataclasses import dataclass
from typing import Any, List

import ipywidgets as widgets
from IPython.display import display
from penelope.common.curve_fit import pchip_spline, rolling_average_smoother
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.utility import get_logger

from .displayers import display_bar, display_line, display_table

logger = get_logger()

DEFAULT_SMOOTHERS = [pchip_spline, rolling_average_smoother('nearest', 3)]


@dataclass
class GUI:

    tab = widgets.Tab()
    normalize = widgets.ToggleButton(description="Normalize", icon='check', value=False)
    smooth = widgets.ToggleButton(description="Smooth", icon='check', value=False)
    status = widgets.HTML("")
    output = widgets.Output(layout=widgets.Layout(width='600px', height='200px'))
    words = widgets.Textarea(
        description="",
        rows=4,
        value="och eller hur",
        layout=widgets.Layout(width='600px', height='200px'),
    )
    display_handlers: List[Any] = None

    def layout(self):
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.normalize,
                        self.smooth,
                        self.status,
                    ]
                ),
                widgets.HBox(
                    [
                        self.words,
                        self.output,
                    ]
                ),
                self.tab,
            ]
        )

    def set_displays(self, handlers: List[Any]):

        self.display_handlers = handlers

        self.tab.children = [widgets.Output() for _ in handlers]
        _ = [self.tab.set_title(i, x.NAME) for i, x in enumerate(self.display_handlers)]

        return self

    @property
    def current_display(self):
        return self.display_handlers[self.tab.selected_index]

    @property
    def current_output(self):
        return self.tab.children[self.tab.selected_index]


def display_gui(state, display_widgets=True):

    gui = GUI().set_displays([display_table, display_line, display_bar])  # , display_grid])

    _corpus: VectorizedCorpus = None

    def toggle_normalize(*_):

        nonlocal _corpus

        _corpus = None
        update_plot(*_)

    def update_plot(*_):

        nonlocal _corpus

        if state.corpus is None:

            with gui.output:
                gui.status.value = "Please load a corpus!"

            return

        if _corpus is None or _corpus is not state.corpus:

            with gui.output:
                gui.status.value = "Corpus changed..."

            _corpus = state.corpus
            if gui.normalize.value:
                _corpus = _corpus.normalize()

            gui.tab.children[1].clear_output()
            with gui.tab.children[1]:
                display_line.setup(state, x_ticks=[x for x in _corpus.xs_years()], plot_width=1000, plot_height=500)

        tokens = '\n'.join(gui.words.value.split()).split()
        indices = [_corpus.token2id[token] for token in tokens if token in _corpus.token2id]

        if len(indices) == 0:
            return

        missing_tokens = [token for token in tokens if token not in _corpus.token2id]

        if len(missing_tokens) > 0:
            gui.status.value = f"<b>Not in corpus subset</b>: {' '.join(missing_tokens)}"
            return

        if gui.current_display.NAME != "Line":
            gui.current_output.clear_output()

        with gui.current_output:

            smoothers = DEFAULT_SMOOTHERS if gui.smooth.value else []

            data = gui.current_display.compile(_corpus, indices, smoothers=smoothers)

            state.data = data

            gui.current_display.plot(data, container=state)

    gui.words.observe(update_plot, names='value')
    gui.tab.observe(update_plot, 'selected_index')
    gui.normalize.observe(toggle_normalize, names='value')
    gui.smooth.observe(update_plot, names='value')

    _layout = gui.layout()

    if display_widgets:

        display(_layout)

        update_plot()

    return _layout
