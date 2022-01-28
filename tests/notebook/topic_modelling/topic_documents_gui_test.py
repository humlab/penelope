from penelope.notebook import topic_modelling as tm


def test_create_gui(state: tm.TopicModelContainer):

    gui: tm.BrowseTopicDocumentsGUI = tm.BrowseTopicDocumentsGUI(state=state).setup()

    layout = gui.layout()
    assert layout is not None

    gui.goto_next()
    gui.goto_previous()
    gui.update_handler()
