import glob
import os
from typing import Callable, List

from ipywidgets import Button, Dropdown, HBox, IntSlider, Layout, Output, VBox

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


view = Output()

DISTANCE_METRIC_OPTIONS = [('linear', 0), ('inverse', 1), ('constant', 2)]


class ComputeGUI:
    def __init__(self):
        self.filepath: Dropdown = Dropdown(description='Corpus', options=[], value=None, layout=Layout(width='400px'))
        self.window_size: IntSlider = IntSlider(
            description='Window', min=2, max=40, value=5, layout=Layout(width='250px')
        )
        self.method: Dropdown = Dropdown(
            description='Method', options=['HAL', 'Glove'], value='HAL', layout=Layout(width='200px')
        )
        self.button: Button = Button(
            description='Compute',
            button_style='Success',
            layout=Layout(width='115px', background_color='blue'),
        )
        self.distance_metric: Dropdown = Dropdown(
            description='Dist.f.', options=DISTANCE_METRIC_OPTIONS, value=2, layout=Layout(width='200px')
        )
        self.compute_handler: Callable = None

    @view.capture(clear_output=True)
    def _compute_handler(self, _):

        if self.filepath.value is None:
            return
        self.button.disabled = True
        try:
            self.compute_handler(
                self.filepath.value,
                window_size=self.window_size.value,
                distance_metric=self.distance_metric.value,
                direction_sensitive=False,  # direction_sensitive.value,
                method=self.method.value,
            )
        finally:
            self.button.disabled = False

    def setup(self, corpus_files: List[str], compute_handler: Callable) -> "ComputeGUI":
        self.filepath.options = corpus_files
        self.compute_handler = compute_handler
        self.button.on_click(self._compute_handler)

        return self

    def layout(self):

        layout = VBox(
            [
                HBox(
                    [
                        VBox([self.filepath, self.method]),
                        VBox([self.window_size, self.distance_metric]),
                        VBox(
                            [
                                # self.direction_sensitive,
                                self.button
                            ]
                        ),
                    ]
                ),
                view,
            ]
        )
        return layout


def create_gui(data_folder, corpus_pattern='*.tokenized.zip', compute_handler: Callable = None):

    corpus_files = sorted(glob.glob(os.path.join(data_folder, corpus_pattern)))

    gui = ComputeGUI().setup(corpus_files=corpus_files, compute_handler=compute_handler)

    return gui
