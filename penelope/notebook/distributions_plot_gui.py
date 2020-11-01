import types
from typing import Iterable, List, Sequence

import ipywidgets
import penelope.plot.word_trend_plot as plotter
from bokeh.plotting import show
from IPython.display import display
from penelope.corpus import VectorizedCorpus


def display_gui(x_corpus: VectorizedCorpus, tokens: Iterable[str], n_columns: int = 3):

    tokens = sorted(list(tokens))
    # tokens_map = {token: index for index, token in enumerate(tokens)}
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
        wtokens=ipywidgets.SelectMultiple(options=tokens, value=[], rows=30),
    )

    def update_plot(*_):

        gui.output.clear_output()

        selected_tokens: Sequence[str] = gui.wtokens.value

        if len(selected_tokens) == 0:
            selected_tokens = tokens[: gui.n_count.value]

        indices: List[int] = [x_corpus.token2id[token] for token in selected_tokens]

        with gui.output:
            x_columns: int = n_columns if gui.split.value else None
            p = plotter.yearly_token_distribution_multiple_line_plot(x_corpus, indices, n_columns=x_columns)
            show(p)

    def stepper_clicked(b):

        current_token = gui.wtokens.value[0] if len(gui.wtokens.value) > 0 else tokens[0]
        current_index = tokens.index(current_token)

        if b.description == "<<":
            current_index = max(current_index - gui.n_count.value, 0)

        if b.description == ">>":
            current_index = min(current_index + gui.n_count.value, len(tokens) - gui.n_count.value + 1)

        gui.wtokens.value = tokens[current_index : current_index + gui.n_count.value]

    def split_changed(*_):
        update_plot()

    def token_select_changed(*_):
        update_plot()

    gui.n_count.observe(update_plot, "value")
    gui.split.observe(split_changed, "value")
    gui.wtokens.observe(token_select_changed, "value")
    gui.forward.on_click(stepper_clicked)
    gui.back.on_click(stepper_clicked)

    display(
        ipywidgets.VBox(
            [
                gui.progress,
                ipywidgets.HBox([gui.back, gui.forward, gui.n_count, gui.split]),
                ipywidgets.HBox([gui.wtokens, gui.output]),
            ]
        )
    )
    update_plot()
