from __future__ import annotations

from contextlib import suppress
from os.path import join as jj
from typing import Any, List, Optional

import ipywidgets as w

from penelope import pipeline
from penelope import topic_modelling as tm

from . import mixins as mx
from . import topic_titles_gui
from .model_container import TopicModelContainer


def load_model(
    *,
    corpus_folder: str,
    state: TopicModelContainer,
    model_name: str,
    corpus_config: pipeline.CorpusConfig = None,
    model_infos: list[dict[str, Any]] = None,
    slim: bool = False,
):

    model_infos = model_infos or tm.find_models(corpus_folder)
    model_info = next(x for x in model_infos if x["name"] == model_name)
    filename_fields = corpus_config.text_reader_opts.filename_fields if corpus_config else None
    trained_model: tm.InferredModel = tm.InferredModel.load(model_info["folder"], lazy=True)
    inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(
        folder=jj(corpus_folder, model_info["name"]), filename_fields=filename_fields, slim=slim
    )

    state.update(trained_model=trained_model, inferred_topics=inferred_topics, train_corpus_folder=model_info["folder"])

    topics = inferred_topics.topic_token_overview

    with suppress(BaseException):
        topic_proportions = inferred_topics.calculator.topic_proportions()
        if topic_proportions is not None:
            topics['score'] = topic_proportions

    if topics is None:
        raise ValueError("bug-check: No topic_token_overview in loaded model!")

    topic_titles_gui.display_gui(topics, topic_titles_gui.PandasTopicTitlesGUI)


class LoadGUI(mx.AlertMixIn):
    def __init__(
        self,
        corpus_folder: str,
        state: TopicModelContainer,
        corpus_config: pipeline.CorpusConfig | None = None,
        slim: bool = False,
    ):
        super().__init__()
        self.corpus_folder: str = corpus_folder
        self.state: TopicModelContainer = state
        self.corpus_config: Optional[pipeline.CorpusConfig] = corpus_config
        self.slim: bool = slim
        self._model_name: w.Dropdown = w.Dropdown(description="Model", options=[], layout=dict(width="40%"))

        self._load: w.Button = w.Button(description="Load", button_style="Success", layout=dict(width="80px"))
        self._output: w.Output = w.Output()

        self.model_infos: List[dict] = tm.find_models(self.corpus_folder)
        self.model_names: List[str] = list(x["name"] for x in self.model_infos)
        self.loaded_model_folder: str = None

    def setup(self) -> "LoadGUI":
        self._model_name.options = self.model_names
        self._load.on_click(self.load_handler)
        return self

    def layout(self) -> w.VBox:
        _layout = w.VBox([w.HBox([self._model_name, self._load, self._alert]), w.VBox([self._output])])
        return _layout

    def load_handler(self, *_):
        try:
            if self._model_name.value is None:
                self.alert("🙃 Please specify which model to load.")
                return
            self._output.clear_output()
            try:
                self._load.disabled = True
                with self._output:
                    self.load()
            finally:
                self._load.disabled = False
        except Exception as ex:
            self.warn(f"😡 {ex}")

    def load(self) -> None:
        self.loaded_model_folder = jj(self.corpus_folder, self._model_name.value)
        load_model(
            corpus_folder=self.corpus_folder,
            state=self.state,
            model_name=self._model_name.value,
            corpus_config=self.corpus_config,
            model_infos=self.model_infos,
            slim=self.slim,
        )


def create_load_topic_model_gui(
    corpus_folder: str,
    state: TopicModelContainer,
    corpus_config: Optional[pipeline.CorpusConfig] = None,
    slim: bool = False,
) -> LoadGUI:

    gui: LoadGUI = LoadGUI(
        corpus_folder=corpus_folder,
        state=state,
        corpus_config=corpus_config,
        slim=slim,
    ).setup()

    return gui
