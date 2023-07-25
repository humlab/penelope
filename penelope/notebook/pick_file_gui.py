from glob import glob
from os.path import basename, join
from typing import Any, Literal, Protocol

import ipyfilechooser
import ipywidgets as w
from loguru import logger

from penelope.utility import default_data_folder

from .utility import shorten_filechooser_label

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


class PickedCallback(Protocol):
    def __call__(self, filename: str, sender: 'PickFileGUI', **kwargs) -> None:
        ...


class PickFileGUI:
    def __init__(
        self,
        folder: str,
        pattern: str,
        picked_callback: PickedCallback = None,
        kind: Literal['chooser', 'picker'] = 'chooser',
    ):
        self.folder: str = folder
        self.kind: Literal['chooser', 'picker'] = kind
        self.pattern: str = pattern or '*_vector_data.npz'
        self.picked_callback: PickedCallback = picked_callback
        self._filename_picker: ipyfilechooser.FileChooser | w.Dropdown = None
        self._alert: w.HTML = w.HTML('.')
        self._load_button = w.Button(
            description='Load', button_style='Success', layout=w.Layout(width='115px'), disabled=True
        )
        self.extra_placeholder: w.HBox = w.HBox([])
        self.payload: Any = None

    def load(self):
        self._load_handler({})

    def _load_handler(self, _):
        try:
            self._load_button.disabled = True
            self.info("⌛ Loading data...")
            self.picked_callback(filename=self.filename, sender=self)
            self.info("✔")

        except FileNotFoundError:
            self.warn("👎 Please select a valid file. 👎")
        except (ValueError, Exception) as ex:
            logger.error(ex)
            self.warn(f"‼ ‼ {ex} ‼ ‼</b>")
        finally:
            self._load_button.disabled = False

    def file_select_callback(self, _: ipyfilechooser.FileChooser):
        self._load_button.disabled = False
        self.alert('✔')

    def setup(self):
        if self.kind == 'picker':
            filenames: list[str] = glob(join(self.folder, "**", self.pattern), recursive=True)
            self._filename_picker = w.Dropdown(
                options={basename(f): f for f in filenames},
                description='Corpus file:',
                disabled=False,
            )
            self._load_button.disabled = False
        else:
            self._filename_picker = ipyfilechooser.FileChooser(
                path=self.folder or default_data_folder(),
                filter_pattern=self.pattern,
                title=f'<b>Open file ({self.pattern})</b>',
                show_hidden=False,
                select_default=True,
                use_dir_icons=True,
                show_only_dirs=False,
            )
            shorten_filechooser_label(self._filename_picker, 50)
            self._filename_picker.register_callback(self.file_select_callback)

        self._load_button.on_click(self._load_handler)
        return self

    def layout(self) -> w.CoreWidget:
        ctrls: list[w.CoreWidget] = (
            [w.VBox([self._alert, self._load_button])] if self.kind == 'chooser' else [self._load_button, self._alert]
        )
        return w.VBox([w.HBox([self._filename_picker] + ctrls)] + [self.extra_placeholder])

    def add(self, widget: w.CoreWidget, append: bool = False):
        self.extra_placeholder.children = (list(self.extra_placeholder.children) if append else []) + [widget]

    @property
    def filename(self):
        if self.kind == 'picker':
            return self._filename_picker.value
        return self._filename_picker.selected

    def alert(self, msg: str):
        self._alert.value = msg

    def warn(self, msg: str):
        self.alert(f"<span style='color=red'>{msg}</span>")

    def info(self, msg: str) -> None:
        self._alert.value = f"<span style='color: green; font-weight: bold;'>{msg or '😃'}</span>"
