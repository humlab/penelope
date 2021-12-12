import pandas as pd
from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.topic_modelling import filter_topic_tokens_overview


def test_filter_topic_tokens_overview(state: TopicModelContainer):

    format_string: str = '#{}#'
    topic_token_overview: pd.DataFrame = state.inferred_topics.topic_token_overview

    data: pd.DataFrame = filter_topic_tokens_overview(
        topic_token_overview, search_text='och', n_count=1, format_string=format_string
    )
    assert data['tokens'].to_csv(sep=';') == 'topic_id;tokens\n0;#och# valv i av sig\n1;#och# som av i en\n'

    data: pd.DataFrame = filter_topic_tokens_overview(
        topic_token_overview, search_text='och', n_count=1, truncate_tokens=True, format_string=format_string
    )
    assert data['tokens'].to_csv(sep=';') == 'topic_id;tokens\n0;#och#\n1;#och#\n'

    data: pd.DataFrame = filter_topic_tokens_overview(
        topic_token_overview, search_text='en', n_count=5, truncate_tokens=True, format_string=format_string
    )
    assert (
        data['tokens'].to_csv(sep=';')
        == 'topic_id;tokens\n1;och som av i #en#\n2;som #en# är i och\n3;de är i #en# som\n'
    )