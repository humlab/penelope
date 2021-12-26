import pandas as pd
import pytest
from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.topic_modelling import (
    EngineKey,
    EngineSpec,
    filter_topic_tokens_overview,
    get_engine_cls_by_method_name,
    get_engine_module_by_method_name,
)


def test_filter_topic_tokens_overview(state: TopicModelContainer):

    format_string: str = '#{}#'
    topic_token_overview: pd.DataFrame = state.inferred_topics.topic_token_overview

    data: pd.DataFrame = filter_topic_tokens_overview(
        topic_token_overview, search_text='och', n_top=1, format_string=format_string
    )
    assert data['tokens'].to_csv(sep=';') == 'topic_id;tokens\n0;#och# valv i av sig\n1;#och# som av i en\n'

    data: pd.DataFrame = filter_topic_tokens_overview(
        topic_token_overview, search_text='och', n_top=1, truncate_tokens=True, format_string=format_string
    )
    assert data['tokens'].to_csv(sep=';') == 'topic_id;tokens\n0;#och#\n1;#och#\n'

    data: pd.DataFrame = filter_topic_tokens_overview(
        topic_token_overview, search_text='en', n_top=5, truncate_tokens=True, format_string=format_string
    )
    assert (
        data['tokens'].to_csv(sep=';')
        == 'topic_id;tokens\n1;och som av i #en#\n2;som #en# är i och\n3;de är i #en# som\n'
    )


@pytest.mark.parametrize('engine_key', ['gensim_lda-multicore', 'gensim_mallet-lda'])
def test_get_engine_cls_by_method_name(engine_key: EngineKey):

    module = get_engine_module_by_method_name(engine_key)
    assert module is not None

    cls = get_engine_cls_by_method_name(engine_key)
    assert cls is not None

    engine_spec: EngineSpec = module.options.get_engine_specification(engine_key=engine_key)

    assert engine_spec.engine is not None
    assert engine_spec.get_options(None, None, dict()) is not None
