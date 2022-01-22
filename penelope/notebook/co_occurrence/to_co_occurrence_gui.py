import contextlib
from typing import Callable, Set, Union

import ipywidgets as widgets

from penelope.co_occurrence import Bundle, ContextOpts
from penelope.pipeline import CorpusConfig
from penelope.workflows import interface

from ..gui_base import BaseGUI, button_layout, default_layout

tooltips = {
    '_context_width': "Max distance to the midmost word, window size two times this value plus one",
    '_ignore_concept': "If checked, the concept words (if specified) are filtered out",
    '_concept': "If specified, then only windows having a focus word in the middle are considered.",
}
view = widgets.Output(layout={"border": "1px solid black"})


class ComputeGUI(BaseGUI):
    def __init__(
        self,
        *,
        default_corpus_path: str = None,
        default_corpus_filename: str = '',
        default_data_folder: str = None,
    ):

        super().__init__(default_corpus_path, default_corpus_filename, default_data_folder)

        self._partition_key: widgets.Dropdown = widgets.Dropdown(
            description='',
            options={'Year': 'year', 'Document': 'document_id'},
            value='document_id',
            layout=widgets.Layout(width='140px'),
            disabled=True,
        )
        self._context_width = widgets.IntSlider(
            description='',
            min=1,
            max=40,
            step=1,
            value=1,
            layout=default_layout,
            # tooltip=tooltips['_context_width'],
        )
        self._concept = widgets.Text(
            value='',
            placeholder='Use comma (,) as word delimiter',
            description='',
            disabled=False,
            layout=widgets.Layout(width='280px'),
            # tooltip=tooltips['_concept'],
        )
        self._ignore_concept = widgets.ToggleButton(
            value=False,
            description='Ignore concept',
            icon='',
            layout=button_layout,
            tooltips="Remove word-pairs that include concept tokens from end result",
        )
        self._ignore_padding = widgets.ToggleButton(
            value=True,
            description='Ignore padding',
            icon='check',
            layout=button_layout,
            tooltips="Remove word-pairs that include padding tokens from end result",
        )
        self._context_width_title = widgets.HTML("<b>Context distance</b>")

    def layout(self, hide_input=False, hide_output=False):

        placeholder: widgets.VBox = self.extra_placeholder
        extra_layout = widgets.HBox(
            [
                widgets.VBox(
                    [
                        self._context_width_title,
                        self._context_width,
                        self._ignore_padding,
                    ]
                ),
                widgets.VBox(
                    [
                        widgets.HTML("<b>Concept</b>"),
                        self._concept,
                        widgets.HBox([self._ignore_concept]),
                    ]
                ),
                widgets.VBox(
                    [
                        widgets.HTML("<b>Pivot</b>"),
                        self._partition_key,
                    ]
                ),
            ]
        )
        placeholder.children = [extra_layout]
        layout = super().layout(hide_input, hide_output)
        return layout

    def setup(self, *, config: CorpusConfig, compute_callback: Callable, done_callback: Callable):
        super().setup(config=config, compute_callback=compute_callback, done_callback=done_callback)
        self._ignore_concept.observe(self._toggle_state_changed, 'value')
        self._ignore_padding.observe(self._toggle_state_changed, 'value')
        self._context_width.observe(self._context_width_changed, 'value')
        self._context_width.value = 2
        return self

    def _context_width_changed(self, _):
        with contextlib.suppress(Exception):
            w: int = 2 * self._context_width.value + 1
            self._context_width_title.value = f"<b>Context distance (w = {w})</b>"

    @property
    def context_opts(self) -> ContextOpts:
        return ContextOpts(
            concept=self.concept_tokens,
            context_width=self._context_width.value,
            ignore_concept=self._ignore_concept.value,
            ignore_padding=self._ignore_padding.value,
            partition_keys=[self._partition_key.value],
        )

    @property
    def concept_tokens(self) -> Set[str]:
        _concepts_str = [x.strip() for x in self._concept.value.strip().split(',') if len(x.strip()) > 1]
        if len(_concepts_str) == 0:
            return set()
        return set(_concepts_str)

    @property
    def compute_opts(self) -> interface.ComputeOpts:
        args: interface.ComputeOpts = super().compute_opts
        args.context_opts = self.context_opts
        return args


def create_compute_gui(
    *,
    corpus_folder: str,
    data_folder: str,
    corpus_config: Union[str, CorpusConfig],
    compute_callback: Callable[[interface.ComputeOpts, CorpusConfig], Bundle] = None,
    done_callback: Callable[[Bundle, interface.ComputeOpts], None] = None,
) -> "ComputeGUI":
    """Returns a GUI for turning a corpus pipeline to co-occurrence data"""
    corpus_config: CorpusConfig = CorpusConfig.find(corpus_config, corpus_folder).folders(corpus_folder)
    gui = ComputeGUI(
        default_corpus_path=corpus_folder,
        default_corpus_filename=(corpus_config.pipeline_payload.source or ''),
        default_data_folder=data_folder,
    ).setup(
        config=corpus_config,
        compute_callback=lambda args, cfg: compute_callback(
            args=args,
            corpus_config=cfg,
        ),
        done_callback=done_callback,
    )

    return gui
