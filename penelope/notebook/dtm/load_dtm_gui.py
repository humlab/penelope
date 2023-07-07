from glob import glob
from os.path import basename, isfile, join, split, splitext
from typing import Callable, Literal

import ipyfilechooser
from ipywidgets import HTML, Button, Dropdown, HBox, Layout, Output, VBox

from penelope.utility import default_data_folder, getLogger, right_chop

from ..utility import shorten_filechooser_label

logger = getLogger('penelope')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


debug_view = Output(layout={'border': '1px solid black'})


class LoadGUI:
    def __init__(
        self,
        folder: str,
        done_callback: Callable[[str, str], None] = None,
        kind: Literal['chooser', 'picker'] = 'chooser',
        filename_pattern: str = None,
    ):
        self.folder: str = folder
        self.kind: Literal['chooser', 'picker'] = kind
        self.filename_pattern: str = filename_pattern or '*_vector_data.npz'
        self.done_callback: Callable[[str, str], None] = done_callback
        self._corpus_filename: ipyfilechooser.FileChooser | Dropdown = None
        self._alert: HTML = HTML('.')
        self._load_button = Button(
            description='Load', button_style='Success', layout=Layout(width='115px'), disabled=True
        )
        self.extra_placeholder: HBox = VBox([])

    def load(self):
        self._load_handler({})

    @debug_view.capture(clear_output=True)
    def _load_handler(self, _):
        try:
            if not self.is_dtm_corpus(self.corpus_filename):
                self.warn("👎 Please select a valid corpus file 👎")
                return

            self._load_button.disabled = True
            folder, filename = split(self.corpus_filename)
            tag = right_chop(filename, self.filename_pattern[1:])
            self.info("⌛ Loading data...")
            self.done_callback(folder=folder, tag=tag)
            self.info("✔")

        except (ValueError, FileNotFoundError, Exception) as ex:
            logger.error(ex)
            self.warn(f"‼ ‼ {ex} ‼ ‼</b>")
        finally:
            self.warn('✔')
            self._load_button.disabled = False

    def is_dtm_corpus(self, filename: str) -> bool:
        if not filename or not isfile(self.corpus_filename):
            return False
        if not splitext(filename)[1] in [".pickle", ".npz"]:
            return False
        return True

    def file_select_callback(self, _: ipyfilechooser.FileChooser):
        self._load_button.disabled = False
        self.alert('✔')

    def setup(self):
        if self.kind == 'picker':
            filenames: list[str] = glob(join(self.folder, "**", self.filename_pattern), recursive=True)
            self._corpus_filename = Dropdown(
                options={basename(f): f for f in filenames},
                description='Corpus file:',
                disabled=False,
            )
        else:
            self._corpus_filename = ipyfilechooser.FileChooser(
                path=self.folder or default_data_folder(),
                filter_pattern=self.filename_pattern,
                title=f'<b>Corpus file ({self.filename_pattern} pickle or npz)</b>',
                show_hidden=False,
                select_default=True,
                use_dir_icons=True,
                show_only_dirs=False,
            )
            shorten_filechooser_label(self._corpus_filename, 50)
            self._corpus_filename.register_callback(self.file_select_callback)

        self._load_button.on_click(self._load_handler)
        return self

    def layout(self):
        ctrls = [VBox([self._alert, self._load_button])] if self.kind == 'chooser' else [self._load_button, self._alert]
        return VBox([HBox([self._corpus_filename] + ctrls)] + [self.extra_placeholder])

    @property
    def corpus_filename(self):
        if self.kind == 'picker':
            return self._corpus_filename.value
        return self._corpus_filename.selected

    def alert(self, msg: str):
        self._alert.value = msg

    def warn(self, msg: str):
        self.alert(f"<span style='color=red'>{msg}</span>")

    def info(self, msg: str) -> None:
        self._alert.value = f"<span style='color: green; font-weight: bold;'>{msg or '😃'}</span>"
