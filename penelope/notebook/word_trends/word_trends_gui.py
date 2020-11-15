from dataclasses import dataclass
from typing import Sequence

import ipywidgets as widgets
from IPython.display import display
from penelope.common.curve_fit import pchip_spline, rolling_average_smoother
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.utility import get_logger

from .displayers import DISPLAYERS, ITrendDisplayer, WordTrendData

logger = get_logger()

DEFAULT_SMOOTHERS = [pchip_spline, rolling_average_smoother('nearest', 3)]
BUTTON_LAYOUT = widgets.Layout(width='80px')
OUTPUT_LAYOUT = widgets.Layout(width='600px')


@dataclass
class GUI:

    tab = widgets.Tab()
    normalize = widgets.ToggleButton(description="Normalize", icon='check', value=False, layout=BUTTON_LAYOUT)
    smooth = widgets.ToggleButton(description="Smooth", icon='check', value=False, layout=BUTTON_LAYOUT)
    status = widgets.HTML(value="", layout=widgets.Layout(width='300px'))
    output = widgets.Output(layout=widgets.Layout(width='600px', height='200px'))
    words = widgets.Textarea(
        description="",
        rows=4,
        value="",
        layout=widgets.Layout(width='600px', height='200px'),
    )
    displayers: Sequence[ITrendDisplayer] = None

    def layout(self):
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.normalize,
                        self.smooth,
                    ]
                ),
                widgets.HBox(
                    [
                        self.words,
                        widgets.VBox(
                            [
                                self.status,
                                self.output,
                            ]
                        ),
                    ]
                ),
                self.tab,
            ]
        )

    def set_displayers(self, *, displayers: Sequence[ITrendDisplayer], trend_data: WordTrendData):

        self.displayers = [cls(trend_data) for cls in displayers]
        self.tab.children = [d.output for d in self.displayers]
        _ = [self.tab.set_title(i, x.name) for i, x in enumerate(self.displayers)]

        return self

    @property
    def current_displayer(self):
        return self.displayers[self.tab.selected_index]

    @property
    def current_output(self):
        return self.current_displayer.output


def word_trend_gui(trend_data: WordTrendData, display_widgets: bool = True) -> widgets.Widget:

    gui = GUI().set_displayers(displayers=DISPLAYERS, trend_data=trend_data)

    _corpus: VectorizedCorpus = None

    def toggle_normalize(*_):

        nonlocal _corpus
        _corpus = None
        update_plot(*_)

    def update_plot(*_):

        nonlocal _corpus

        if trend_data.corpus is None:
            gui.status.value = "Please load a corpus!"
            return

        if _corpus is None or _corpus is not trend_data.corpus:

            gui.status.value = "Corpus changed..."

            _corpus = trend_data.corpus
            if gui.normalize.value:
                _corpus = _corpus.normalize()

            for displayer in gui.displayers:
                displayer.setup()

        tokens = '\n'.join(gui.words.value.split()).split()
        indices = [_corpus.token2id[token] for token in tokens if token in _corpus.token2id]

        if len(indices) == 0:
            return

        missing_tokens = [token for token in tokens if token not in _corpus.token2id]

        if len(missing_tokens) > 0:
            gui.status.value = f"<b>Not found</b>: {' '.join(missing_tokens)}"
            return

        gui.current_displayer.clear()

        smoothers = DEFAULT_SMOOTHERS if gui.smooth.value else []

        data = gui.current_displayer.compile(_corpus, indices, smoothers=smoothers)

        gui.current_displayer.plot(data, state=trend_data)

    gui.words.observe(update_plot, names='value')
    gui.tab.observe(update_plot, 'selected_index')
    gui.normalize.observe(toggle_normalize, names='value')
    gui.smooth.observe(update_plot, names='value')

    _layout = gui.layout()

    if display_widgets:

        display(_layout)

        update_plot()

    return _layout
