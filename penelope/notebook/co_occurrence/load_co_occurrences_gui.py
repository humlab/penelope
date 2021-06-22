import os
from typing import Callable

import penelope.co_occurrence as co_occurrence
from ipywidgets import HTML, Button, HBox, Layout, Output, VBox
from loguru import logger
from penelope.utility import default_data_folder

from ..utility import CLEAR_OUTPUT, FileChooserExt2

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes

debug_view = Output(layout={"border": "1px solid black"})


class LoadGUI:
    def __init__(self, default_path: str = None, default_filename: str = '', filename_pattern: str = None):

        self.default_path: str = default_path
        self.default_filename: str = default_filename
        self.filename_pattern: str = filename_pattern

        self._filename: FileChooserExt2 = None
        self._load_button: Button = Button(
            description='Load', button_style='Success', layout=Layout(width='120px'), disabled=True
        )

        self.load_callback: Callable[[str], None] = None
        self.loaded_callback: Callable[[co_occurrence.Bundle], None] = None

    # @view.capture(clear_output=True)
    def _load_handler(self, *_):

        if self.load_callback is None:
            return

        if self._load_button.disabled:
            return

        self._load_button.description = "Loading"
        self._load_button.disabled = True
        self._filename.disabled = True
        try:
            self.info("⌛ Loading bundle...")
            bundle = self.load_callback(self.filename)
            self.info("✔ Bundle loaded!")
            if self.loaded_callback is not None:
                self.info("⌛ Preparing display...")
                self.loaded_callback(bundle)
                self.info("✔")
        except (ValueError, FileNotFoundError) as ex:
            self.alert("😮 Load failed!")
            print(ex)
        except Exception as ex:
            self.alert("😮 Load failed!")
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
        self._alert: HTML = HTML("📌")
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
                            [self._alert, self._load_button],
                            layout=Layout(display='flex', align_items='flex-end', justify_content='space-between'),
                        ),
                    ]
                ),
                debug_view,
            ]
        )

    def alert(self, message: str) -> None:
        self._alert.value = f"<span style='color: red; font-weight: bold;'>{message or '🙁'}</span>"

    def info(self, message: str) -> None:
        self._alert.value = f"<span style='color: green; font-weight: bold;'>{message or '😃'}</span>"

    @property
    def filename(self):
        return self._filename.selected


@debug_view.capture(clear_output=CLEAR_OUTPUT)
def create_load_gui(
    data_folder: str,
    filename_pattern: str = co_occurrence.FILENAME_PATTERN,
    loaded_callback: Callable[[co_occurrence.Bundle], None] = None,
) -> "LoadGUI":

    gui: LoadGUI = LoadGUI(default_path=data_folder).setup(
        filename_pattern=filename_pattern,
        load_callback=load_co_occurrence_bundle,
        loaded_callback=loaded_callback,
    )

    return gui


@debug_view.capture(clear_output=CLEAR_OUTPUT)
def load_co_occurrence_bundle(filename: str) -> co_occurrence.Bundle:
    try:
        if not filename or not os.path.isfile(filename):
            raise ValueError("Please select co-occurrence file")

        bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename)

        return bundle
    except (ValueError, FileNotFoundError, PermissionError) as ex:
        logger.error(ex)
        raise
    except Exception as ex:
        logger.error(ex)
        raise
