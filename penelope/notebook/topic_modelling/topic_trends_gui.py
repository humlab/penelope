import types
import warnings

import ipywidgets as widgets
import penelope.topic_modelling as topic_modelling
from IPython.display import display

from .. import widgets_utils
from .display_topic_trends import display_topic_trends
from .model_container import TopicModelContainer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def display_gui(state: TopicModelContainer):

    text_id = 'topic_share_plot'

    weighings = [(x['description'], x['key']) for x in topic_modelling.YEARLY_MEAN_COMPUTE_METHODS]

    gui = types.SimpleNamespace(
        n_topics=state.num_topics,
        text_id=text_id,
        text=widgets_utils.text_widget(text_id),
        aggregate=widgets.Dropdown(
            description='Aggregate', options=weighings, value='true_mean', layout=widgets.Layout(width="200px")
        ),
        normalize=widgets.ToggleButton(description='Normalize', value=True, layout=widgets.Layout(width="120px")),
        topic_id=widgets.IntSlider(
            description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0, continuous_update=False
        ),
        output_format=widgets.Dropdown(
            description='Format', options=['Chart', 'Table'], value='Chart', layout=widgets.Layout(width="200px")
        ),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0),
        output=widgets.Output(),
        prev_topic_id=None,
        next_topic_id=None,
    )

    gui.prev_topic_id = widgets_utils.button_with_previous_callback(gui, 'topic_id', state.num_topics)
    gui.next_topic_id = widgets_utils.button_with_next_callback(gui, 'topic_id', state.num_topics)

    def on_topic_change_update_gui(topic_id):

        if gui.n_topics != state.num_topics:
            gui.n_topics = state.num_topics
            gui.topic_id.value = 0
            gui.topic_id.max = state.num_topics - 1

        tokens = topic_modelling.get_topic_title(state.inferred_topics.topic_token_weights, topic_id, n_tokens=200)

        gui.text.value = 'ID {}: {}'.format(topic_id, tokens)

    def update_handler(*_):

        gui.output.clear_output()

        with gui.output:

            on_topic_change_update_gui(gui.topic_id.value)

            weights = topic_modelling.compute_topic_yearly_means(state.inferred_topics.document_topic_weights)

            display_topic_trends(
                weight_over_time=weights,
                topic_id=gui.topic_id.value,
                year_range=state.inferred_topics.year_period,
                aggregate=gui.aggregate.value,
                normalize=gui.normalize.value,
                output_format=gui.output_format.value,
            )

    gui.topic_id.observe(update_handler, names='value')
    gui.normalize.observe(update_handler, names='value')
    gui.aggregate.observe(update_handler, names='value')
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
                        widgets.VBox([gui.topic_id]),
                        widgets.VBox([gui.aggregate, gui.output_format]),
                        widgets.VBox([gui.normalize]),
                    ]
                ),
                gui.text,
                gui.output,
            ]
        )
    )

    update_handler()
