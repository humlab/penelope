from __future__ import annotations

from contextlib import suppress
from os.path import join as jj
from typing import Any, List

import ipywidgets as w
import pandas as pd
from IPython.display import display
from loguru import logger

from penelope import pipeline as pp
from penelope import topic_modelling as tm

from . import mixins as mx
from . import topic_titles_gui as tt_ui
from .model_container import TopicModelContainer


def load_model(
    *,
    corpus_folder: str,
    state: TopicModelContainer,
    model_name: str,
    model_infos: list[dict[str, Any]] = None,
    slim: bool = False,
    n_tokens: int = 500,
):

    model_infos = model_infos or tm.find_models(corpus_folder)
    model_info = next(x for x in model_infos if x["name"] == model_name)

    corpus_config: pp.CorpusConfig = pp.CorpusConfig.find("corpus.yml", corpus_folder)
    filename_fields = corpus_config.text_reader_opts.filename_fields if corpus_config else None
    trained_model: tm.InferredModel = tm.InferredModel.load(model_info["folder"], lazy=True)

    if corpus_config is None:
        logger.warning("no corpus config found in model folder")

    inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(
        folder=jj(corpus_folder, model_info["name"]), filename_fields=filename_fields, slim=slim
    )

    state.update(trained_model=trained_model, inferred_topics=inferred_topics, train_corpus_folder=model_info["folder"])

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
        corpus_folder: str,
        state: TopicModelContainer,
        slim: bool = False,
    ):
        super().__init__()
        self.corpus_folder: str = corpus_folder
        self.state: TopicModelContainer = state
        self.slim: bool = slim
        self._model_name: w.Dropdown = w.Dropdown(description="Model", options=[], layout=dict(width="40%"))

        self._load: w.Button = w.Button(description="Load", button_style="Success", layout=dict(width="100px"))
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
            model_infos=self.model_infos,
            slim=self.slim,
        )
