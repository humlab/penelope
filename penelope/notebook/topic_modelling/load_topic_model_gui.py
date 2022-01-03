import warnings
from contextlib import suppress
from os.path import join as jj
from typing import Any, Callable, Dict, List, Optional

from ipywidgets import Button, Dropdown, HBox, Layout, Output, VBox  # type: ignore
from penelope import pipeline
from penelope.topic_modelling import InferredModel, InferredTopicsData, find_models

from . import topic_titles_gui
from .model_container import TopicModelContainer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_model(
    corpus_config: pipeline.CorpusConfig,
    corpus_folder: str,
    state: TopicModelContainer,
    model_name: str,
    model_infos: List[Dict[str, Any]] = None,
):

    model_infos = model_infos or find_models(corpus_folder)
    model_info = next(x for x in model_infos if x["name"] == model_name)
    filename_fields = corpus_config.text_reader_opts.filename_fields if corpus_config else None
    inferred_model: InferredModel = InferredModel.load(model_info["folder"], lazy=True)
    inferred_topics: InferredTopicsData = InferredTopicsData.load(
        folder=jj(corpus_folder, model_info["name"]),
        filename_fields=filename_fields,
    )

    state.set_data(inferred_model, inferred_topics, _train_corpus_folder=model_info["folder"])

    topics = inferred_topics.topic_token_overview

    with suppress(BaseException):
        topic_proportions = inferred_topics.compute_topic_proportions()
        if topic_proportions is not None:
            topics['score'] = topic_proportions

    # topics.style.set_properties(**{'text-align': 'left'}).set_table_styles(
    #     [dict(selector='td', props=[('text-align', 'left')])]
    # )

    if topics is None:
        raise ValueError("bug-check: No topic_token_overview in loaded model!")

    topic_titles_gui.display_gui(topics, topic_titles_gui.PandasTopicTitlesGUI)


class LoadGUI:
    def __init__(self):
        self.model_name: Dropdown = Dropdown(description="Model", options=[], layout=Layout(width="40%"))
        self.load: Button = Button(description="Load", button_style="Success", layout=Layout(width="80px"))
        self.output: Output = Output()
        self.load_callback: Callable = None

    def setup(self, model_names: List[str], load_callback: Callable = None) -> "LoadGUI":
        self.model_name.options = model_names
        self.load_callback = load_callback
        self.load.on_click(self.load_handler)
        return self

    def layout(self) -> VBox:
        _layout = VBox([HBox([self.model_name, self.load]), VBox([self.output])])
        return _layout

    def load_handler(self, *_):

        if self.model_name.value is None:
            print("Please specify which model to load.")
            return

        self.output.clear_output()
        try:
            self.load.disabled = True
            with self.output:
                self.load_callback(self.model_name.value)
        finally:
            self.load.disabled = False


def create_load_topic_model_gui(
    corpus_config: Optional[pipeline.CorpusConfig], corpus_folder: str, state: TopicModelContainer
) -> LoadGUI:

    model_infos: List[dict] = find_models(corpus_folder)
    model_names: List[str] = list(x["name"] for x in model_infos)

    def load_callback(model_name: str):
        load_model(corpus_config, corpus_folder, state, model_name, model_infos)

    gui = LoadGUI().setup(model_names, load_callback=load_callback)

    return gui
