import os

import penelope.pipeline.checkpoint as checkpoint
from penelope.corpus.readers.interfaces import ExtractTaggedTokensOpts
from penelope.pipeline.convert import tagged_frame_to_tokens


def test_spaCy_load_tagged_frame_checkpoint():
    """Loads CSV files stored in a ZIP as Pandas data frames. """

    os.makedirs('./tests/output', exist_ok=True)

    checkpoint_filename: str = "./tests/test_data/SSI_tagged_frame_pos_csv.zip"

    checkpoint_opts: checkpoint.CheckpointOpts = None  # CheckpointOpts()
    data = checkpoint.load_checkpoint(source_name=checkpoint_filename, checkpoint_opts=checkpoint_opts, reader_opts=None)

    assert data is not None


def test_phrased_tagged_frame():

    os.makedirs('./tests/output', exist_ok=True)

    checkpoint_filename: str = "./tests/test_data/SSI_tagged_frame_pos_csv.zip"
    checkpoint_opts: checkpoint.CheckpointOpts = None  # CheckpointOpts()
    data = checkpoint.load_checkpoint(source_name=checkpoint_filename, checkpoint_opts=checkpoint_opts, reader_opts=None)
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

    phrases = {'United': 'United', 'United Nations': 'United_Nations'}
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

    phrases = {'United Nations': 'United_Nations', 'the United Nations': 'the_United_Nations'}

    phrased_tokens = tagged_frame_to_tokens(
        payload.content,
        ExtractTaggedTokensOpts(lemmatize=False),
        phrases=phrases,
    )
    assert phrased_tokens[:8] == [
        'Constitution',
        'of',
        'the_United_Nations',
        'Educational',
        ',',
        'Scientific',
        'and',
        'Cultural',
    ]

    phrases = {'united nations': 'United_Nations'}

    phrased_tokens = tagged_frame_to_tokens(
        payload.content,
        ExtractTaggedTokensOpts(lemmatize=False),
        phrases=phrases,
        ignore_case=True,
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


def test_python_list_merge():

    tokens = [chr(ord('a') + i) for i in range(0, 10)]

    tokens[3:6] = ['_'.join(tokens[3:6])]

    assert tokens == ['a', 'b', 'c', 'd_e_f', 'g', 'h', 'i', 'j']

    phrases = [['d', 'e', 'f']]

    phrases_strs = [' '.join(p) for p in phrases]

    tokens_str = ' '.join(tokens)

    for s in phrases_strs:
        tokens_str = tokens_str.replace(s, s.replace(' ', '_'))

    tokens = tokens_str.split()

    assert tokens == ['a', 'b', 'c', 'd_e_f', 'g', 'h', 'i', 'j']
