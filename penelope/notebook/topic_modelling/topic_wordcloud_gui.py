import types

import ipywidgets as widgets
import pandas as pd
import penelope.plot as plot_utility
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from IPython.display import display

from .. import widgets_utils
from .model_container import TopicModelContainer

pd.set_option("display.max_rows", None)
pd.set_option('max_colwidth', 200)

opts = {'max_font_size': 100, 'background_color': 'white', 'width': 900, 'height': 600}


def display_wordcloud(state: TopicModelContainer, topic_id=0, n_words=100, output_format='Wordcloud', gui=None):
    def tick(n=None):
        gui.progress.value = (gui.progress.value + 1) if n is None else n

    if gui.n_topics != state.num_topics:
        gui.n_topics = state.num_topics
        gui.topic_id.value = 0
        gui.topic_id.max = state.num_topics - 1

    tick(1)

    try:
        topic_token_weights = state.inferred_topics.topic_token_weights

        df = topic_token_weights.loc[(topic_token_weights.topic_id == topic_id)]

        tokens = topic_modelling.get_topic_title(topic_token_weights, topic_id, n_tokens=n_words)

        if len(tokens) == 0:
            print("No data! Please change selection.")
            return

        gui.text.value = 'ID {}: {}'.format(topic_id, tokens)

        tick()

        if output_format == 'Wordcloud':
            plot_utility.plot_wordcloud(df, token='token', weight='weight', max_words=n_words, **opts)
        else:
            df = topic_modelling.get_topic_tokens(topic_token_weights, topic_id=topic_id, n_tokens=n_words)
            if output_format == 'Table':
                display(df)
            if output_format == 'Excel':
                filename = utility.timestamp("{}_wordcloud_tokens.xlsx")
                df.to_excel(filename)
            if output_format == 'CSV':
                filename = utility.timestamp("{}_wordcloud_tokens.csv")
                df.to_csv(filename, sep='\t')
    except IndexError:
        print('No data for topic')
    tick(0)


def display_gui(state: TopicModelContainer):

    output_options = ['Wordcloud', 'Table', 'CSV', 'Excel']
    text_id = 'tx02'

    gui = types.SimpleNamespace(
        n_topics=state.num_topics,
        text_id=text_id,
        text=widgets.HTML(value=f"<span class='{text_id}'></span>", placeholder='', description=''),
        topic_id=widgets.IntSlider(
            description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0, continuous_update=False
        ),
        word_count=widgets.IntSlider(description='#Words', min=5, max=250, step=1, value=25, continuous_update=False),
        output_format=widgets.Dropdown(
            description='Format', options=output_options, value=output_options[0], layout=widgets.Layout(width="200px")
        ),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0, layout=widgets.Layout(width="95%")),
        prev_topic_id=None,
        next_topic_id=None,
    )

    gui.prev_topic_id = widgets_utils.button_with_previous_callback(gui, 'topic_id', state.num_topics)
    gui.next_topic_id = widgets_utils.button_with_next_callback(gui, 'topic_id', state.num_topics)

    iw = widgets.interactive(
        display_wordcloud,
        state=widgets.fixed(state),
        topic_id=gui.topic_id,
        n_words=gui.word_count,
        output_format=gui.output_format,
        gui=widgets.fixed(gui),
    )

    display(
        widgets.VBox(
            [
                gui.text,
                widgets.HBox([gui.prev_topic_id, gui.next_topic_id, gui.topic_id, gui.word_count, gui.output_format]),
                gui.progress,
                iw.children[-1],
            ]
        )
    )

    iw.update()
