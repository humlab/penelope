import types

import ipywidgets as widgets
import penelope.notebook.widgets_utils as widgets_utils
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from IPython.display import display

from .display_topic_trends_heatmap import display_heatmap
from .model_container import TopicModelContainer

logger = utility.get_logger()


def display_gui(state: TopicModelContainer):

    lw = lambda w: widgets.Layout(width=w)

    text_id = 'topic_relevance'

    # year_min, year_max = state.inferred_topics.year_period

    titles = topic_modelling.get_topic_titles(state.inferred_topics.topic_token_weights, n_tokens=100)
    weighings = [(x['description'], x['key']) for x in topic_modelling.YEARLY_MEAN_COMPUTE_METHODS]

    gui = types.SimpleNamespace(
        text_id=text_id,
        text=widgets_utils.text_widget(text_id),
        flip_axis=widgets.ToggleButton(value=False, description='Flip', icon='', layout=lw("80px")),
        aggregate=widgets.Dropdown(
            description='Aggregate', options=weighings, value='max_weight', layout=widgets.Layout(width="250px")
        ),
        output_format=widgets.Dropdown(
            description='Output', options=['Heatmap', 'Table'], value='Heatmap', layout=lw("180px")
        ),
        output=widgets.Output(),
    )

    def update_handler(*_):

        gui.output.clear_output()

        with gui.output:

            weights = topic_modelling.compute_topic_yearly_means(state.inferred_topics.document_topic_weights).fillna(0)

            display_heatmap(
                weights,
                titles,
                flip_axis=gui.flip_axis.value,
                aggregate=gui.aggregate.value,
                output_format=gui.output_format.value,
            )

    gui.aggregate.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')
    gui.flip_axis.observe(update_handler, names='value')

    display(
        widgets.VBox(
            [widgets.HBox([gui.aggregate, gui.output_format, gui.flip_axis]), widgets.HBox([gui.output]), gui.text]
        )
    )

    update_handler()
