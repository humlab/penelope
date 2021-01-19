import penelope.co_occurrence as co_occurrence
import penelope.notebook.co_occurrence as co_occurrences_gui

DATA_FOLDER = './tests/test_data'


def test_create_load_co_occurrences_gui():
    def load_callback(_: str) -> co_occurrence.Bundle:
        return None

    def loaded_callback(_: co_occurrence.Bundle):
        ...

    gui = co_occurrences_gui.create_load_gui(data_folder=DATA_FOLDER)

    gui = gui.setup(
        filename_pattern=co_occurrence.CO_OCCURRENCE_FILENAME_PATTERN,
        load_callback=load_callback,
        loaded_callback=loaded_callback,
    )
    assert gui is not None

    layout = gui.layout()

    assert layout is not None
