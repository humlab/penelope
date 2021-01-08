import os

import penelope.pipeline.checkpoint as checkpoint


def test_spaCy_load_tagged_frame_checkpoint():
    """Loads CSV files stored in a ZIP as Pandas data frames. """

    os.makedirs('./tests/output', exist_ok=True)

    checkpoint_filename: str = "./tests/test_data/SSI_tagged_frame_pos_csv.zip"

    options: checkpoint.CorpusSerializeOpts = None  # CorpusSerializeOpts()
    data = checkpoint.load_checkpoint(source_name=checkpoint_filename, options=options, reader_opts=None)

    assert data is not None
