

from penelope import co_occurrence
from penelope.notebook.co_occurrence import CoOccurrenceTable


def test_co_occurrence_table_create():

    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename = co_occurrence.to_filename(folder=folder, tag=tag)

    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_frame=False)

    assert bundle is not None


    gui = CoOccurrenceTable(
        co_occurrences=bundle.co_occurrences,
        token2id=bundle.token2id,
        document_index=bundle.document_index,
        concepts=set(),
    )

    assert gui is not None
