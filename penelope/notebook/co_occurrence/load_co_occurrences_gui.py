import os
from dataclasses import dataclass, field
from typing import Callable

import penelope.co_occurrence as co_occurrence
from ipywidgets import Button, HBox, Label, Layout, Output, VBox
from penelope.notebook.utility import FileChooserExt2
from penelope.utility import default_data_folder, getLogger

logger = getLogger('penelope')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


debug_view = Output(layout={"border": "1px solid black"})


@dataclass
class LoadGUI:

    default_path: str = field(default=None)
    default_filename: str = field(default='')
    filename_pattern: str = None

    _filename: FileChooserExt2 = None
    _load_button: Button = Button(
        description='Load', button_style='Success', layout=Layout(width='120px'), disabled=True
    )

    load_callback: Callable[[str], None] = None
    loaded_callback: Callable[[co_occurrence.Bundle], None] = None

    # @view.capture(clear_output=True)
    def _load_handler(self, *_):

        if self.load_callback is None:
            return

        self._load_button.description = "Loading"
        self._load_button.disabled = True
        self._filename.disabled = True
        try:
            bundle = self.load_callback(self.filename)
            if self.loaded_callback is not None:
                self.loaded_callback(bundle)
        except (ValueError, FileNotFoundError) as ex:
            print(ex)
        except Exception as ex:
            logger.info(ex)
            raise
        finally:
            self._load_button.disabled = False
            self._filename.disabled = False
            self._load_button.description = "Load"

    def file_selected(self, *_):
        self._load_button.disabled = not self._filename.selected

    def setup(
        self,
        filename_pattern: str,
        load_callback: Callable[[str], None],
        loaded_callback: Callable[[co_occurrence.Bundle], None],
    ) -> "LoadGUI":  # pylint: disable=redefined-outer-name)

        self.filename_pattern = filename_pattern

        self._filename = FileChooserExt2(
            path=self.default_path or default_data_folder(),
            filename=self.default_filename or '',
            filter_pattern=filename_pattern or '*.*',
            title=f'<b>Co-occurrence file</b> ({filename_pattern})',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False,
        )

        self._filename.register_callback(self.file_selected)
        self._load_button.on_click(self._load_handler)
        self.load_callback = load_callback
        self.loaded_callback = loaded_callback

        return self

    def layout(self):
        return VBox(
            [
                HBox(
                    [
                        VBox([self._filename]),
                        VBox(
                            [Label("λ"), self._load_button],
                            layout=Layout(display='flex', align_items='flex-end', justify_content='space-between'),
                        ),
                    ]
                ),
                debug_view,
            ]
        )

    @property
    def filename(self):
        return self._filename.selected


@debug_view.capture(clear_output=True)
def create_load_gui(
    data_folder: str,
    filename_pattern: str = co_occurrence.CO_OCCURRENCE_FILENAME_PATTERN,
    loaded_callback: Callable[[co_occurrence.Bundle], None] = None,
) -> "LoadGUI":

    gui: LoadGUI = LoadGUI(default_path=data_folder).setup(
        filename_pattern=filename_pattern,
        load_callback=load_co_occurrence_bundle,
        loaded_callback=loaded_callback,
    )

    return gui


@debug_view.capture(clear_output=True)
def load_co_occurrence_bundle(filename: str) -> co_occurrence.Bundle:
    try:
        if not filename or not os.path.isfile(filename):
            raise ValueError("Please select co-occurrence file")

        bundle = co_occurrence.load_bundle(filename)
        return bundle
    except (ValueError, FileNotFoundError, PermissionError) as ex:
        logger.error(ex)
        raise
    except Exception as ex:
        logger.error(ex)
        raise
