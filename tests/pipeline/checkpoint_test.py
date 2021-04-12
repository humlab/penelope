import os
from io import StringIO

import pandas as pd
import penelope.pipeline.checkpoint as checkpoint
from penelope.corpus.readers.interfaces import ExtractTaggedTokensOpts, TextReaderOpts
from penelope.pipeline.convert import tagged_frame_to_tokens


def test_load_tagged_frame_checkpoint():
    """Loads CSV files stored in a ZIP as Pandas data frames. """

    os.makedirs('./tests/output', exist_ok=True)

    checkpoint_filename: str = "./tests/test_data/legal_instrument_five_docs_test_pos_csv.zip"

    checkpoint_opts: checkpoint.CheckpointOpts = None  # CheckpointOpts()
    data = checkpoint.load_checkpoint(
        source_name=checkpoint_filename, checkpoint_opts=checkpoint_opts, reader_opts=None
    )

    assert data is not None

    assert data.source_name == "legal_instrument_five_docs_test_pos_csv.zip"
    assert len(data.document_index) == 5

    payloads = [x for x in data.payload_stream]
    assert len(payloads) == 5
    assert all(x.filename in data.document_index.filename.to_list() for x in payloads)

    """Test reader_opts filter by list of filenames"""
    whitelist = {'RECOMMENDATION_0201_049455_2017.txt', 'DECLARATION_0201_013178_1997.txt'}
    data = checkpoint.load_checkpoint(
        source_name=checkpoint_filename,
        checkpoint_opts=checkpoint_opts,
        reader_opts=TextReaderOpts(filename_filter=whitelist),
    )
    assert {x.filename for x in data.payload_stream} == whitelist
    assert {x for x in data.document_index.filename.to_list()} == whitelist

    """Test reader_opts filter by a predicate"""
    years = [1958, 1945]
    expected_documents = {'CONSTITUTION_0201_015244_1945_london.txt', 'CONVENTION_0201_015395_1958_paris.txt'}
    whitelister = lambda x: x.split('_')[3] in map(str, years)
    data = checkpoint.load_checkpoint(
        source_name=checkpoint_filename,
        checkpoint_opts=checkpoint_opts,
        reader_opts=TextReaderOpts(filename_filter=whitelister),
    )
    assert {x.filename for x in data.payload_stream} == expected_documents
    assert {x for x in data.document_index.filename.to_list()} == expected_documents


def test_phrased_tagged_frame():

    os.makedirs('./tests/output', exist_ok=True)
    opts = dict(
        extract_opts=ExtractTaggedTokensOpts(lemmatize=False),
        filter_opts=None,
        text_column='text',
        lemma_column='lemma_',
        pos_column='pos_',
        ignore_case=False,
        verbose=True,
    )
    data_str: str = """	text	lemma_	pos_	is_punct	is_stop
0	Constitution	constitution	NOUN	False	False
1	of	of	ADP	False	True
2	the	the	DET	False	True
3	United	United	PROPN	False	False
4	Nations	Nations	PROPN	False	False
5	Educational	Educational	PROPN	False	False
6	,	,	PUNCT	True	False
7	Scientific	Scientific	PROPN	False	False
8	and	and	CCONJ	False	True
9	Cultural	Cultural	PROPN	False	False
10	Organization	Organization	PROPN	False	False"""

    tagged_frame: pd.date_range1 = pd.read_csv(StringIO(data_str), sep='\t', index_col=0)

    tokens = tagged_frame_to_tokens(tagged_frame, **opts)
    assert tokens is not None

    phrases = {'United Nations': 'United_Nations', 'United': 'United'}
    phrased_tokens = tagged_frame_to_tokens(tagged_frame, **opts, phrases=phrases)
    assert phrased_tokens[:9] == 'Constitution of the United_Nations Educational , Scientific and Cultural'.split(' ')

    phrases = {'United Nations': 'United_Nations', 'the United Nations': 'the_United_Nations'}
    phrased_tokens = tagged_frame_to_tokens(tagged_frame, **opts, phrases=phrases)
    assert phrased_tokens[:8] == 'Constitution of the_United_Nations Educational , Scientific and Cultural'.split(' ')

    phrases = {'united nations': 'United_Nations'}

    phrased_tokens = tagged_frame_to_tokens(tagged_frame, phrases=phrases, **{**opts, **{'ignore_case': True}})
    assert phrased_tokens[:9] == 'Constitution of the United_Nations Educational , Scientific and Cultural'.split(' ')


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
