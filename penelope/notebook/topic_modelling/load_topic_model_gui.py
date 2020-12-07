import types
import warnings
from os.path import join as jj
from typing import Any, Dict, List

import ipywidgets as widgets
import numpy as np
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from IPython.display import display
from penelope.corpus import add_document_index_attributes
from penelope.topic_modelling.container import InferredModel, InferredTopicsData

from . import display_topic_titles
from .display_topic_titles import DisplayPandasGUI
from .model_container import TopicModelContainer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = utility.get_logger()


# FIXME: #94 Column 'year' is missing in `document_index` in model metadata (InferredTopicsData)
def temporary_bug_fixupdate_documents(inferred_topics):

    logger.info("applying temporary bug fix of missing year in document_index...done!")
    document_index = inferred_topics.document_index
    document_topic_weights = inferred_topics.document_topic_weights

    if "year" not in document_index.columns:
        document_index["year"] = document_index.filename.str.split("_").apply(lambda x: x[1]).astype(np.int)

    if "year" not in document_topic_weights.columns:
        document_topic_weights = add_document_index_attributes(catalogue=document_index, target=document_topic_weights)

    inferred_topics.document_index = document_index
    inferred_topics.document_topic_weights = document_topic_weights

    assert "year" in inferred_topics.document_index.columns
    assert "year" in inferred_topics.document_topic_weights.columns

    return inferred_topics


def load_model(
    corpus_folder: str,
    state: TopicModelContainer,
    model_name: str,
    model_infos: List[Dict[str, Any]] = None,
):

    model_infos = model_infos or topic_modelling.find_models(corpus_folder)
    model_info = next(x for x in model_infos if x["name"] == model_name)

    inferred_model: InferredModel = topic_modelling.load_model(model_info["folder"], lazy=True)
    inferred_topics: InferredTopicsData = topic_modelling.InferredTopicsData.load(jj(corpus_folder, model_info["name"]))

    inferred_topics = temporary_bug_fixupdate_documents(inferred_topics)

    state.set_data(inferred_model, inferred_topics)

    topics = inferred_topics.topic_token_overview
    # topics.style.set_properties(**{'text-align': 'left'}).set_table_styles(
    #     [dict(selector='td', props=[('text-align', 'left')])]
    # )

    if topics is None:
        raise ValueError("bug-check: No topic_token_overview in loaded model!")

    display_topic_titles.display_gui(topics, DisplayPandasGUI)


@utility.try_catch
def display_gui(corpus_folder: str, state: TopicModelContainer):

    model_infos = topic_modelling.find_models(corpus_folder)
    model_names = list(x["name"] for x in model_infos)

    gui = types.SimpleNamespace(
        model_name=widgets.Dropdown(description="Model", options=model_names, layout=widgets.Layout(width="40%")),
        load=widgets.Button(
            description="Load",
            button_style="Success",
            layout=widgets.Layout(width="80px"),
        ),
        output=widgets.Output(),
    )

    def load_handler(*_):  # pylint: disable=unused-argument
        gui.output.clear_output()
        try:
            gui.load.disabled = True
            with gui.output:
                if gui.model_name.value is None:
                    print("Please specify which model to load.")
                    return
                load_model(corpus_folder, state, gui.model_name.value, model_infos)
        finally:
            gui.load.disabled = False

    gui.load.on_click(load_handler)

    display(widgets.VBox([widgets.HBox([gui.model_name, gui.load]), widgets.VBox([gui.output])]))
