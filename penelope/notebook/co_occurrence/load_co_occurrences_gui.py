import os
from dataclasses import dataclass, field
from typing import Callable

from ipywidgets import Button, HBox, Label, Layout, VBox  # , Output
from penelope.co_occurrence import load_co_occurrences, to_vectorized_corpus
from penelope.co_occurrence.interface import CO_OCCURRENCE_FILENAME_POSTFIX
from penelope.corpus import VectorizedCorpus
from penelope.notebook.utility import FileChooserExt2
from penelope.utility import default_data_folder, getLogger, read_json, replace_extension, right_chop

logger = getLogger('penelope')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


# view = Output(layout={"border": "1px solid black"})


@dataclass
class GUI:

    default_path: str = field(default=None)
    default_filename: str = field(default='')
    filename_pattern: str = None

    _filename: FileChooserExt2 = None
    _load_button: Button = Button(
        description='Load', button_style='Success', layout=Layout(width='120px'), disabled=True
    )

    load_callback: Callable = None

    # @view.capture(clear_output=True)
    def _load_handler(self, *_):

        if self.load_callback is None:
            return

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

    def setup(self, filename_pattern: str, load_callback: str) -> "GUI":  # pylint: disable=redefined-outer-name)

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
                # view,
            ]
        )

    @property
    def filename(self):
        return self.filename.selected


# @view.capture(clear_output=True)
def load_callback(
    args: GUI,
    loaded_callback: Callable,
):

    try:

        if not args.filename or not os.path.isfile(args.filename):
            raise ValueError("Please select co-occurrence file")

        co_occurrences = load_co_occurrences(args.filename)

        options_filename = replace_extension(args.filename, 'json')
        if os.path.isfile(options_filename):
            compute_options = read_json(options_filename)
        else:
            logger.warning(f"options file {options_filename} not found")
            compute_options = {'not_found': options_filename}

        corpus_folder = os.path.split(args.filename)
        corpus_tag = right_chop(args.filename, CO_OCCURRENCE_FILENAME_POSTFIX)

        corpus = (
            VectorizedCorpus.load(folder=corpus_folder, tag=corpus_tag)
            if VectorizedCorpus.dump_exists(folder=corpus_folder, tag=corpus_tag)
            else to_vectorized_corpus(
                co_occurrences=co_occurrences,
                value_column='value_n_t',
            ).group_by_year()
        )

        if loaded_callback is not None:
            loaded_callback(
                corpus_folder=corpus_folder,
                corpus_tag=corpus_tag,
                co_occurrences=co_occurrences,
                corpus=corpus,
                compute_options=compute_options,
            )

    except (ValueError, FileNotFoundError, PermissionError) as ex:
        logger.error(ex)
    except Exception as ex:
        logger.error(ex)


def create_gui(
    data_folder: str,
    filename_pattern: str = "*.co_occurrence.csv.zip",
    loaded_callback: Callable[..., None] = None,
):

    gui = GUI(default_path=data_folder).setup(
        filename_pattern=filename_pattern,
        load_callback=lambda args: load_callback(args, loaded_callback),
    )

    return gui
