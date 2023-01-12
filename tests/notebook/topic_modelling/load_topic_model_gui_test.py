from penelope.notebook.topic_modelling.load_topic_model_gui import LoadGUI


def test_load_gui(state):
    data_folder: str = './tests/test_data'
    gui: LoadGUI = LoadGUI(data_folder=data_folder, state=state, slim=False).setup()

    layout = gui.layout()
    assert layout is not None
    gui._model_name.value = gui._model_name.options[0]  # pylint: disable=protected-access
    gui.load()
