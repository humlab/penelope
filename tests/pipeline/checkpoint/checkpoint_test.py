import os

import pytest

import penelope.pipeline.checkpoint as checkpoint
from penelope.corpus import TextReaderOpts


@pytest.mark.long_running
def test_load_tagged_frame_checkpoint():
    """Loads CSV files stored in a ZIP as Pandas data frames."""

    os.makedirs('./tests/output', exist_ok=True)

    tagged_corpus_source: str = "./tests/test_data/SSI/legal_instrument_five_docs_test_pos_csv.zip"

    checkpoint_opts: checkpoint.CheckpointOpts = None  # CheckpointOpts()
    data = checkpoint.load_archive(source_name=tagged_corpus_source, checkpoint_opts=checkpoint_opts, reader_opts=None)

    assert data is not None

    assert os.path.basename(data.source_name) == "legal_instrument_five_docs_test_pos_csv.zip"
    assert len(data.document_index) == 5

    payloads = [x for x in data.create_stream()]
    assert len(payloads) == 5
    assert all(x.filename in data.document_index.filename.to_list() for x in payloads)

    """Test reader_opts filter by list of filenames"""
    whitelist = {'RECOMMENDATION_0201_049455_2017.txt', 'DECLARATION_0201_013178_1997.txt'}
    data = checkpoint.load_archive(
        source_name=tagged_corpus_source,
        checkpoint_opts=checkpoint_opts,
        reader_opts=TextReaderOpts(filename_filter=whitelist),
    )
    assert {x.filename for x in data.create_stream()} == whitelist
    assert {x for x in data.document_index.filename.to_list()} == whitelist

    """Test reader_opts filter by a predicate"""
    years = [1958, 1945]
    expected_documents = {'CONSTITUTION_0201_015244_1945_london.txt', 'CONVENTION_0201_015395_1958_paris.txt'}
    whitelister = lambda x: x.split('_')[3] in map(str, years)
    data = checkpoint.load_archive(
        source_name=tagged_corpus_source,
        checkpoint_opts=checkpoint_opts,
        reader_opts=TextReaderOpts(filename_filter=whitelister),
    )
    assert {x.filename for x in data.create_stream()} == expected_documents
    assert {x for x in data.document_index.filename.to_list()} == expected_documents


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
