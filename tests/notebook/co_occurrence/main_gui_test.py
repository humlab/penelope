from penelope import co_occurrence
from penelope.notebook.co_occurrence import main_gui


def test_to_trends_data():

    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename = co_occurrence.to_filename(folder=folder, tag=tag)

    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_frame=False)

    trends_data = main_gui.to_trends_data(bundle).update()

    assert trends_data is not None
