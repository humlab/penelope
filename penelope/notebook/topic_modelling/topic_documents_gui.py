import types
import warnings

import ipywidgets as widgets
import penelope.notebook.widgets_utils as widgets_utils
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from IPython.display import display

from .model_container import TopicModelContainer
from .utility import filter_document_topic_weights

logger = utility.get_logger()
# from beakerx import *
# from beakerx.object import beakerx
# beakerx.pandas_display_table()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def display_documents(
    inferred_topics: topic_modelling.InferredTopicsData, filters, threshold=0.0, output_format='Table', n_top=500
):

    document_topic_weights = filter_document_topic_weights(
        inferred_topics.document_topic_weights, filters=filters, threshold=threshold
    )

    if len(document_topic_weights) == 0:
        print('No data to display for this topic and threshold')
        return

    if output_format == 'Table':
        document_topic_weights = (
            document_topic_weights.drop(['topic_id'], axis=1)
            .set_index('document_id')
            .sort_values('weight', ascending=False)
            .head(n_top)
        )
        document_topic_weights.index.name = 'id'
        display(document_topic_weights)


def display_gui(state: TopicModelContainer):

    text_id = 'topic_document_text'

    gui = types.SimpleNamespace(
        n_topics=state.num_topics,
        text_id=text_id,
        text=widgets_utils.text_widget(text_id),
        topic_id=widgets.IntSlider(
            description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0, continuous_update=False
        ),
        n_top=widgets.IntSlider(description='#Docs', min=5, max=500, step=1, value=75),
        threshold=widgets.FloatSlider(
            description='Threshold', min=0.0, max=1.0, step=0.01, value=0.20, continues_update=False
        ),
        output_format=widgets.Dropdown(
            description='Format', options=['Table'], value='Table', layout=widgets.Layout(width="200px")
        ),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0),
        output=widgets.Output(),
        prev_topic_id=None,
        next_topic_id=None,
    )

    gui.prev_topic_id = widgets_utils.button_with_previous_callback(gui, 'topic_id', state.num_topics)
    gui.next_topic_id = widgets_utils.button_with_next_callback(gui, 'topic_id', state.num_topics)

    def on_topic_change_update_gui(topic_id: int):

        topic_token_weights = state.inferred_topics.topic_token_weights
        if gui.n_topics != state.num_topics:
            gui.n_topics = state.num_topics
            gui.topic_id.value = 0
            gui.topic_id.max = state.num_topics - 1

        if len(topic_token_weights[topic_token_weights.topic_id == topic_id]) == 0:
            tokens = ["Topics has no significant presence in any documents in the entire corpus"]
        else:
            tokens = topic_modelling.get_topic_title(topic_token_weights, topic_id, n_tokens=200)

        gui.text.value = 'ID {}: {}'.format(topic_id, tokens)

    def update_handler(*_):

        gui.output.clear_output()

        with gui.output:

            on_topic_change_update_gui(gui.topic_id.value)

            display_documents(
                inferred_topics=state.inferred_topics,
                filters=dict(topic_id=gui.topic_id.value),
                threshold=gui.threshold.value,
                n_top=gui.n_top.value,
                output_format=gui.output_format.value,
            )

    gui.topic_id.observe(update_handler, names='value')
    gui.threshold.observe(update_handler, names='value')
    gui.n_top.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(
        widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HBox([gui.prev_topic_id, gui.next_topic_id]),
                                gui.progress,
                            ]
                        ),
                        widgets.VBox([gui.topic_id, gui.threshold, gui.n_top]),
                        widgets.VBox([gui.output_format]),
                    ]
                ),
                gui.text,
                gui.output,
            ]
        )
    )

    update_handler()
