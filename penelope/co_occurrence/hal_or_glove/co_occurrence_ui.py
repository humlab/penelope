import glob
import os
from dataclasses import dataclass
from typing import Callable, List

import ipywidgets
import penelope.utility as utility

logger = utility.getLogger('penelope')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


view = ipywidgets.Output()


@dataclass
class ComputeGUI:

    distance_metric_options = [('linear', 0), ('inverse', 1), ('constant', 2)]

    filepath = ipywidgets.Dropdown(
        description='Corpus', options=[], value=None, layout=ipywidgets.Layout(width='400px')
    )
    window_size = ipywidgets.IntSlider(
        description='Window', min=2, max=40, value=5, layout=ipywidgets.Layout(width='250px')
    )
    method = ipywidgets.Dropdown(
        description='Method', options=['HAL', 'Glove'], value='HAL', layout=ipywidgets.Layout(width='200px')
    )
    button = ipywidgets.Button(
        description='Compute',
        button_style='Success',
        layout=ipywidgets.Layout(width='115px', background_color='blue'),
    )

    distance_metric = ipywidgets.Dropdown(
        description='Dist.f.', options=distance_metric_options, value=2, layout=ipywidgets.Layout(width='200px')
    )

    compute_handler: Callable = None

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

        layout = ipywidgets.VBox(
            [
                ipywidgets.HBox(
                    [
                        ipywidgets.VBox([self.filepath, self.method]),
                        ipywidgets.VBox([self.window_size, self.distance_metric]),
                        ipywidgets.VBox(
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
