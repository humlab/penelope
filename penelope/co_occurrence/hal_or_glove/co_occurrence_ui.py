import glob
import os

import ipywidgets
import penelope.utility as utility

logger = utility.getLogger('corpus_text_analysis')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


class CoccurrenceUI:
    def __init__(self, data_folder, corpus_pattern='*.tokenized.zip'):

        self.data_folder = data_folder
        self.corpus_pattern = corpus_pattern

    def build(self, compute_handler):
        def on_button_clicked(_):

            if self.filepath.value is None:
                return

            self.out.clear_output()
            with self.out:
                self.button.disabled = True
                compute_handler(
                    self.filepath.value,
                    window_size=self.window_size.value,
                    distance_metric=self.distance_metric.value,
                    direction_sensitive=False,  # self.direction_sensitive.value,
                    method=self.method.value,
                )
                self.button.disabled = False

        corpus_files = sorted(glob.glob(os.path.join(self.data_folder, self.corpus_pattern)))
        distance_metric_options = [('linear', 0), ('inverse', 1), ('constant', 2)]

        self.filepath = ipywidgets.Dropdown(
            description='Corpus', options=corpus_files, value=None, layout=ipywidgets.Layout(width='400px')
        )
        self.window_size = ipywidgets.IntSlider(
            description='Window', min=2, max=40, value=5, layout=ipywidgets.Layout(width='250px')
        )
        self.method = ipywidgets.Dropdown(
            description='Method', options=['HAL', 'Glove'], value='HAL', layout=ipywidgets.Layout(width='200px')
        )
        self.button = ipywidgets.Button(
            description='Compute',
            button_style='Success',
            layout=ipywidgets.Layout(width='115px', background_color='blue'),
        )
        self.out = ipywidgets.Output()

        self.distance_metric = ipywidgets.Dropdown(
            description='Dist.f.', options=distance_metric_options, value=2, layout=ipywidgets.Layout(width='200px')
        )
        # self.direction_sensitive = widgets.ToggleButton(description='L/R', value=False, layout=widgets.Layout(width='115px',background_color='blue'))
        # self.zero_diagonal       = widgets.ToggleButton(description='Zero Diag', value=False, layout=widgets.Layout(width='115px',background_color='blue'))

        self.button.on_click(on_button_clicked)

        return ipywidgets.VBox(
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
                self.out,
            ]
        )
