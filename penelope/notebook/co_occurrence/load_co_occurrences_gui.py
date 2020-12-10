import os
from dataclasses import dataclass, field
from typing import Callable

import ipyfilechooser
import ipywidgets as widgets
from penelope.co_occurrence import load_co_occurrences
from penelope.utility import default_data_folder, getLogger, read_json, replace_extension, right_chop, strip_paths

logger = getLogger('penelope')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes

col_layout = widgets.Layout(width='400px')
button_layout = widgets.Layout(width='120px')


@dataclass
class GUI:

    default_path: str = field(default=None)
    default_filename: str = field(default='')
    filename_pattern: str = None

    _filename: ipyfilechooser.FileChooser = None
    _load_button: widgets.Button = widgets.Button(
        description='Load', button_style='Success', layout=button_layout, disabled=True
    )

    output: widgets.Output = widgets.Output()

    load_callback: Callable = None

    def _load_handler(self, *_):

        if self.load_callback is None:
            return

        self.output.clear_output()

        with self.output:
            self._load_button.disabled = True
            try:
                self.load_callback(self)
            except (ValueError, FileNotFoundError) as ex:
                print(ex)
            except Exception as ex:
                logger.info(ex)
                raise
            finally:
                self._load_button.disabled = False

    def file_selected(self, *_):
        self._load_button.disabled = not self._filename.selected

    def setup(self, filename_pattern: str, load_callback: str) -> "GUI":
        self.filename_pattern = filename_pattern
        self.filename = ipyfilechooser.FileChooser(
            path=self.default_path or default_data_folder(),
            filename=self.default_filename,
            filter_pattern=filename_pattern,
            title=f'<b>Co-occurrence file</b> ({filename_pattern})',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False,
        )

        self._filename.register_callback(self.file_selected)
        self._load_button.on_click(self._load_handler)
        self.load_callback = load_callback
        return self

    def layout(self):
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                self._filename,
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.Label(""),
                                self._load_button,
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

    @property
    def filename(self):
        return self.filename.selected


def _load_co_occurrence(args: GUI, loaded_callback: Callable):

    try:
        if not args.filename:
            raise ValueError("Please select co-occurrence file")

        output_tag = right_chop(strip_paths(args.filename), args.filename_pattern.replace('*', ''))

        if output_tag == "":
            raise ValueError("Filename's tag cannot be determined")

        if not os.path.isfile(args.filename):
            raise FileNotFoundError(args.filename)

        co_occurrences = load_co_occurrences(args.filename)
        compute_options = read_json(replace_extension(args.filename, 'json'))

        if loaded_callback is not None:
            loaded_callback(
                concept_co_occurrences=co_occurrences,
                compute_options=compute_options,
            )

    except (ValueError, FileNotFoundError, PermissionError) as ex:
        logger.error(ex)
    except Exception as ex:
        logger.error(ex)


def create_gui(
    data_folder: str,
    filename_pattern: str = ".coo_concept_context.csv.zip",
    loaded_callback: Callable[..., None] = None,
):

    gui = GUI(default_path=data_folder).setup(
        filename_pattern=filename_pattern,
        load_callback=lambda args: _load_co_occurrence(args, loaded_callback),
    )

    return gui
