from __future__ import annotations

from contextlib import suppress
from typing import Any

import ipywidgets as w
import pandas as pd
from IPython.display import display
from loguru import logger

from penelope import pipeline as pp
from penelope import topic_modelling as tm

from . import mixins as mx
from . import topic_titles_gui as tt_ui
from .model_container import TopicModelContainer


def _get_filename_fields(folder: str) -> Any:
    try:
        corpus_configs: list[pp.CorpusConfig] = pp.CorpusConfig.find_all(folder)
        if len(corpus_configs) > 0:
            return corpus_configs[0].text_reader_opts.filename_fields
    except FileNotFoundError:
        logger.warning(f"corpus.yml not found in model folder! Please copy config file to {folder}.")
    return None


def load_model(*, state: TopicModelContainer, model_info: tm.ModelFolder, slim: bool = False, n_tokens: int = 500):

    trained_model: tm.InferredModel = tm.InferredModel.load(model_info.folder, lazy=True)

    inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(
        folder=model_info.folder, filename_fields=_get_filename_fields(model_info.folder), slim=slim
    )

    state.update(trained_model=trained_model, inferred_topics=inferred_topics, train_corpus_folder=model_info.folder)

    topics: pd.DataFrame = inferred_topics.topic_token_overview
    topics['tokens'] = inferred_topics.get_topic_titles(n_tokens=n_tokens)

    columns_to_show: list[str] = [column for column in ['tokens', 'alpha', 'coherence'] if column in topics.columns]

    topics = topics[columns_to_show]

    with suppress(BaseException):
        topic_proportions = inferred_topics.calculator.topic_proportions()
        if topic_proportions is not None:
            topics['score'] = topic_proportions

    if topics is None:
        raise ValueError("bug-check: No topic_token_overview in loaded model!")

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
                with self._output:
                    self.load()
            finally:
                self._load.disabled = False
        except Exception as ex:
            self.warn(f"😡 {ex}")

    def load(self) -> None:
        load_model(state=self.state, model_info=self.model_info, slim=self.slim)

    @property
    def model_info(self) -> tm.ModelFolder | None:
        return next((m for m in self.model_infos if m.name == self.model_name), None)

    @property
    def model_name(self) -> str | None:
        return self._model_name.value
