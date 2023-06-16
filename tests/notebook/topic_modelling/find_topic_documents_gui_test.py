import pandas as pd

from penelope import topic_modelling as tm
from penelope.common import render_text as rt
from penelope.corpus import load_document_index
from penelope.notebook import topic_modelling as ntm

# pylint: disable=protected-access


def test_filter_topic_tokens_overview(inferred_topics_data: tm.InferredTopicsData):
    topics: pd.DataFrame = inferred_topics_data.topic_token_overview
    topics['tokens'] = inferred_topics_data.get_topic_titles(n_tokens=200)

    reduced_topics = tm.filter_topic_tokens_overview(topics, search_text='Valv', n_top=50, truncate_tokens=True)

    assert reduced_topics is not None
    assert len(reduced_topics) > 0

    reduced_topics = tm.filter_topic_tokens_overview(
        topics, search_text='DettaOrdFinnsInte', n_top=50, truncate_tokens=True
    )

    assert reduced_topics is not None
    assert len(reduced_topics) == 0


def test_find_topic_documents_gui(state):
    gui: ntm.FindTopicDocumentsGUI = ntm.FindTopicDocumentsGUI(state=state).setup()

    layout = gui.layout()
    assert layout is not None

    gui.update_handler()


def test_find_topic_documents(state: ntm.TopicModelContainer):
    text_repository: rt.ITextRepository = rt.TextRepository(
        source='tests/test_data/tranströmer_corpus.zip',
        document_index=state.inferred_topics.document_index_proper,
    )

    render_service: rt.IRenderService = rt.RenderService(template='{{text}}', links_registry={})

    gui: ntm.WithPivotKeysText.FindTopicDocumentsGUI = ntm.WithPivotKeysText.FindTopicDocumentsGUI(
        state=state,
        text_repository=text_repository,
        render_service=render_service,
    )
    gui.setup()

    layout = gui.layout()
    assert layout is not None

    document_name: str = 'tran_2019_03_test'

    expected_text: str = text_repository.get_text(f'{document_name}.txt')

    gui.on_row_click(item={'document_name': document_name}, g=None)

    assert gui.text_output.value == expected_text

    gui.render_service.template = '{{document_id}} {{document_name}}: {{text}}'
    gui.on_row_click(item={'document_name': document_name}, g=None)

    assert gui.text_output.value == f"2 {document_name}: {expected_text}"

    gui._find_text.value = 'Valv'
    gui.update_handler()
