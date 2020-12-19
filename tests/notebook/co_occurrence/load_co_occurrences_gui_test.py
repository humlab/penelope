import penelope.notebook.co_occurrence.load_co_occurrences_gui as load_co_occurrences_gui

DATA_FOLDER = './tests/test_data'


def test_load_co_occurrences_gui_create_gui():
    def load_callback(_: load_co_occurrences_gui.GUI):
        pass

    gui = load_co_occurrences_gui.create_gui(data_folder=DATA_FOLDER)
    gui = gui.setup(filename_pattern="*.zip", load_callback=load_callback)
    assert gui is not None

    layout = gui.layout()

    assert layout is not None


# def test_GUI_setup_and_layout():
#     pass

# @patch('VectorizedCorpus')
# def test_load_callback():
#     args: load_co_occurrences_gui.GUI = Mock(spec=load_co_occurrences_gui.GUI)
#     load_co_occurrences_gui.load_callback(args, loaded_callback=loaded_callback)
