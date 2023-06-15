import pandas as pd

from penelope.notebook.topic_modelling import LoadGUI, TopicModelContainer, PandasTopicTitlesGUI


def test_load_gui(state):
    data_folder: str = './tests/test_data'
    gui: LoadGUI = LoadGUI(data_folder=data_folder, state=state, slim=False).setup()

    layout = gui.layout()
    assert layout is not None
    gui._model_name.value = gui._model_name.options[0]  # pylint: disable=protected-access
    gui.load()


def test_pandas_topic_titles_gui(state: TopicModelContainer):
    inferred_topics = state.inferred_topics
    n_tokens = 50
    topics: pd.DataFrame = inferred_topics.topic_token_overview
    topics['tokens'] = inferred_topics.get_topic_titles(n_tokens=n_tokens)

    columns_to_show: list[str] = [column for column in ['tokens', 'alpha', 'coherence'] if column in topics.columns]

    topics = topics[columns_to_show]

    topic_proportions = inferred_topics.calculator.topic_proportions()
    if topic_proportions is not None:
        topics['score'] = topic_proportions

    if topics is None:
        raise ValueError("bug-check: No topic_token_overview in loaded model!")

    tt_gui = PandasTopicTitlesGUI(topics, n_tokens=n_tokens).setup()

    assert tt_gui is not None
