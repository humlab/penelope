from __future__ import annotations

import ipywidgets as w
from IPython.display import display

from penelope import topic_modelling as tm

from . import mixins as mx
from . import topic_titles_gui as tt_ui
from .model_container import TopicModelContainer


def _load_tm_handler(
    *, state: TopicModelContainer, model_info: tm.ModelFolder, slim: bool = False, n_tokens: int = 500
):
    state.load(folder=model_info.folder, slim=slim)

    topics = state.inferred_topics.get_topics_overview_with_score(n_tokens=n_tokens)

    display(tt_ui.PandasTopicTitlesGUI(topics, n_tokens=n_tokens).setup().layout())


class LoadGUI(mx.AlertMixIn):
    def __init__(
        self,
        data_folder: str,
        state: TopicModelContainer,
        slim: bool = False,
    ):
        super().__init__()
        self.data_folder: str = data_folder
        self.state: TopicModelContainer = state
        self.slim: bool = slim
        self._model_name: w.Dropdown = w.Dropdown(description="Model", options=[], layout=dict(width="40%"))

        self._load: w.Button = w.Button(description="Load", button_style="Success", layout=dict(width="100px"))
        self._output: w.Output = w.Output()

        self.model_infos: list[tm.ModelFolder] = tm.find_models(self.data_folder)

    def setup(self) -> "LoadGUI":
        self._model_name.options = [x.name for x in self.model_infos]
        if self.model_name is None and self.model_infos:
            self._model_name.value = self.model_infos[0].name
        self._load.on_click(self.load_handler)
        return self

    def layout(self) -> w.VBox:
        _layout = w.VBox([w.HBox([self._model_name, self._load, self._alert]), w.VBox([self._output])])
        return _layout

    def load_handler(self, *_):
        try:
            if self.model_name is None:
                self.alert("🙃 Please specify which model to load.")
                return
            self._output.clear_output()
            try:
                self._load.disabled = True
                self.alert("🙃 Loading...")
                with self._output:
                    self.load()
            finally:
                self._load.disabled = False
                self.alert("🙃 Done!")
        except Exception as ex:
            self.warn(f"😡 {ex}")

    def load(self) -> None:
        _load_tm_handler(state=self.state, model_info=self.model_info, slim=self.slim)

    @property
    def model_info(self) -> tm.ModelFolder | None:
        return next((m for m in self.model_infos if m.name == self.model_name), None)

    @property
    def model_name(self) -> str | None:
        return self._model_name.value
