import os
from dataclasses import dataclass
from typing import Callable

import ipyfilechooser
import ipywidgets as widgets
from penelope.co_occurrence import load_co_occurrences
from penelope.utility import default_data_folder, getLogger, right_chop
from penelope.utility.file_utility import read_json
from penelope.utility.filename_utils import replace_extension, strip_paths

logger = getLogger('penelope')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


def _layout(width, **kwargs):
    return widgets.Layout(width=width, **kwargs)


@dataclass
class GUI:

    filename_suffix = ".coo_concept_context.csv.zip"
    col_layout = _layout('400px')
    button_layout = _layout('120px')
    input_filename_chooser = ipyfilechooser.FileChooser(
        path=default_data_folder(),
        filter_pattern=f'*{filename_suffix}',
        title=f'<b>Co-occurrence file</b> (*{filename_suffix})',
        show_hidden=False,
        select_default=False,
        use_dir_icons=True,
        show_only_dirs=False,
    )

    button = widgets.Button(description='Load', button_style='Success', layout=button_layout, disabled=True)
    output = widgets.Output()

    def set_filename_suffix(self, path, value):
        self.filename_suffix = value
        self.input_filename_chooser.path = path
        self.input_filename_chooser.filter_pattern = f'*.{self.filename_suffix}'
        self.input_filename_chooser.title = f'<b>Co-occurrence file</b> (*{self.filename_suffix})'

    def layout(self):
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                self.input_filename_chooser,
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.Label(""),
                                self.button,
                            ],
                            layout=widgets.Layout(
                                display='flex',
                                align_items='flex-end',
                                justify_content='space-between',
                            ),
                        ),
                    ]
                ),
                self.output,
            ]
        )


def display_gui(
    data_folder: str, filename_suffix: str = ".coo_concept_context.csv.zip", loaded_callback: Callable[..., None] = None
):

    # Hard coded for now, must be changed!!!!

    gui = GUI()
    gui.set_filename_suffix(data_folder, filename_suffix)

    def on_button_clicked(_):

        try:
            input_filename = gui.input_filename_chooser.selected
            if input_filename is None or input_filename == "":
                raise ValueError("Please select co-occurrence file")

            metadata_filename = replace_extension(input_filename, 'json')
            output_tag = right_chop(strip_paths(input_filename), filename_suffix)

            if output_tag == "":
                raise ValueError("Filename's tag cannot be determined")

            if not os.path.isfile(input_filename):
                raise FileNotFoundError(input_filename)

            gui.output.clear_output()

            with gui.output:

                gui.button.disabled = True

                concept_co_occurrences = load_co_occurrences(input_filename)
                compute_options = read_json(metadata_filename)

                if loaded_callback is not None:
                    loaded_callback(
                        output=gui.output,
                        concept_co_occurrences=concept_co_occurrences,
                        compute_options=compute_options,
                    )

        except (
            ValueError,
            FileNotFoundError,
            PermissionError,
        ) as ex:
            with gui.output:
                logger.error(ex)
        except Exception as ex:
            with gui.output:
                logger.error(ex)

        finally:
            gui.button.disabled = False

    def file_selected(*_):
        with gui.output:
            gui.button.disabled = not gui.input_filename_chooser.selected

    gui.input_filename_chooser.register_callback(file_selected)
    gui.button.on_click(on_button_clicked)

    return gui.layout()
