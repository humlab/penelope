import os

import pytest
from penelope.corpus.readers import ExtractTaggedTokensOpts
from penelope.pipeline import checkpoint
from penelope.pipeline.convert import tagged_frame_to_tokens


@pytest.mark.skip("################# No implemented DONT FORGET #######################")
def test_phrased_tagged_frame():

    os.makedirs('./tests/output', exist_ok=True)

    checkpoint_filename: str = "./tests/test_data/tranströmer_corpus_export.sparv4.csv.zip"
    options: checkpoint.CorpusSerializeOpts = None  # CorpusSerializeOpts()
    data = checkpoint.load_checkpoint(source_name=checkpoint_filename, options=options, reader_opts=None)
    payload = next(data.payload_stream)

    tokens = tagged_frame_to_tokens(
        payload.content,
        ExtractTaggedTokensOpts(lemmatize=False),
    )
    assert tokens is not None
    phrases = {'United Nations': 'United_Nations', 'United': 'United'}
    phrased_tokens = tagged_frame_to_tokens(
        payload.content,
        ExtractTaggedTokensOpts(lemmatize=False),
        phrases=phrases,
    )
    assert phrased_tokens[:9] == [
        'Constitution',
        'of',
        'the',
        'United_Nations',
        'Educational',
        ',',
        'Scientific',
        'and',
        'Cultural',
    ]
