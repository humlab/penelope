import os
from io import StringIO

import pandas as pd
import pytest
from penelope.corpus import ExtractTaggedTokensOpts
from penelope.pipeline.checkpoint import CheckpointOpts
from penelope.pipeline.convert import detect_phrases, merge_phrases, tagged_frame_to_tokens
from penelope.pipeline.sparv import SparvCsvSerializer

# pylint: disable=redefined-outer-name

TEST_CSV_POS_DOCUMENT: str = """token	pos	baseform
# text
Inne	AB	|inne|
i	RG	|
den	PN	|den|
väldiga	JJ	|väldig|
romanska	JJ	|romansk|
kyrkan	NN	|kyrka|
trängdes	VB	|tränga|trängas|
turisterna	NN	|turist|
i	PL	|
halvmörkret	NN	|halvmörker|
.	MAD	|
Valv	NN	|valv|
gapade	VB	|gapa|
bakom	PP	|bakom|
valv	NN	|valv|
och	UO	|
ingen	PN	|ingen|
överblick	NN	|överblick|
.	MAD	|
Några	DT	|någon|
ljuslågor	NN	|ljuslåga|
fladdrade	VB	|fladdra|
.	MAD	|
"""


@pytest.mark.skip(reason="Not implemented")
def test_to_vectorized_corpus():
    pass


def create_checkpoint_opts() -> CheckpointOpts:
    options: CheckpointOpts = CheckpointOpts(
        content_type_code=1,
        document_index_name=None,
        document_index_sep=None,
        sep='\t',
        quoting=3,
        custom_serializer_classname="penelope.pipeline.sparv.convert.SparvCsvSerializer",
        deserialize_in_parallel=False,
        deserialize_processes=4,
        deserialize_chunksize=4,
        text_column="token",
        lemma_column="baseform",
        pos_column="pos",
        extra_columns=[],
        index_column=None,
    )
    return options


@pytest.fixture(scope='module')
def checkpoint_opts() -> CheckpointOpts:
    options: CheckpointOpts = create_checkpoint_opts()
    return options


def create_tagged_frame():
    deserializer: SparvCsvSerializer = SparvCsvSerializer()
    options: CheckpointOpts = create_checkpoint_opts()
    tagged_frame: pd.DataFrame = deserializer.deserialize(content=TEST_CSV_POS_DOCUMENT, options=options)
    return tagged_frame


@pytest.fixture()
def tagged_frame():
    tagged_frame: pd.DataFrame = create_tagged_frame()
    return tagged_frame


# @pytest.mark.parametrize("extract_opts,expected", [(dict(lemmatize=False, pos_includes=None, pos_excludes=None),[])])
def test_tagged_frame_to_tokens_pos_and_lemma(tagged_frame: pd.DataFrame):

    opts = dict(filter_opts=None, text_column='token', lemma_column='baseform', pos_column='pos')

    extract_opts = ExtractTaggedTokensOpts(lemmatize=False, pos_includes=None, pos_excludes=None)
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == tagged_frame.token.tolist()

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes=None, pos_excludes=None)
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == tagged_frame.baseform.tolist()

    extract_opts = ExtractTaggedTokensOpts(lemmatize=False, pos_includes='VB', pos_excludes=None)
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == ['trängdes', 'gapade', 'fladdrade']

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes='VB', pos_excludes=None)
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == ['tränga', 'gapa', 'fladdra']

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes='|VB|', pos_excludes=None)
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == tagged_frame[tagged_frame.pos.isin(['VB'])].baseform.tolist()

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes='|VB|NN|', pos_excludes=None)
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == tagged_frame[tagged_frame.pos.isin(['VB', 'NN'])].baseform.tolist()

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes=None, pos_excludes='MID|MAD|PAD')
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == tagged_frame[~tagged_frame.pos.isin(['MID', 'MAD'])].baseform.tolist()

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes='|VB|', pos_excludes=None)
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == tagged_frame[tagged_frame.pos.isin(['VB'])].baseform.tolist()


def test_tagged_frame_to_tokens_with_passthrough(tagged_frame: pd.DataFrame):

    opts = dict(filter_opts=None, text_column='token', lemma_column='baseform', pos_column='pos')

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=False, pos_includes='VB', pos_excludes=None, passthrough_tokens=['kyrkan', 'ljuslågor']
    )
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == ['kyrkan', 'trängdes', 'gapade', 'ljuslågor', 'fladdrade']


def test_tagged_frame_to_tokens_replace_pos(tagged_frame: pd.DataFrame):

    opts = dict(filter_opts=None, text_column='token', lemma_column='baseform', pos_column='pos')

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes="NN", pos_excludes='MID|MAD|PAD', pos_paddings="VB"
    )
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == ["kyrka", "*", "turist", "halvmörker", "valv", "*", "valv", "överblick", "ljuslåga", "*"]

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes=None, pos_excludes='MID|MAD|PAD', pos_paddings="VB|NN"
    )
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == (
        ['inne', 'i', 'den', 'väldig', 'romansk', '*', '*', '*', 'i', '*', '*', '*']
        + ['bakom', '*', 'och', 'ingen', '*', 'någon', '*', '*']
    )

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes="JJ", pos_excludes='MID|MAD|PAD', pos_paddings="VB|NN"
    )
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == ['väldig', 'romansk', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']


def test_tagged_frame_to_tokens_detect_phrases(tagged_frame: pd.DataFrame):

    opts = dict(filter_opts=None, text_column='token', lemma_column='baseform', pos_column='pos')

    expected_tokens = tagged_frame.baseform[:4].tolist() + ['romansk_kyrka'] + tagged_frame.baseform[6:].tolist()
    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes=None)
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts, phrases=[["romansk", "kyrka"]])
    assert tokens == expected_tokens

    # TODO: Test ignore_case argument


def test_tagged_frame_to_tokens_with_append_pos_true(tagged_frame: pd.DataFrame):

    opts = dict(filter_opts=None, text_column='token', lemma_column='baseform', pos_column='pos')

    extract_opts = ExtractTaggedTokensOpts(lemmatize=False, pos_includes='VB', pos_excludes=None, append_pos=True)
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == ['trängdes@VB', 'gapade@VB', 'fladdrade@VB']

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes="JJ", pos_excludes='MID|MAD|PAD', pos_paddings="VB|NN", append_pos=True
    )
    tokens = tagged_frame_to_tokens(tagged_frame, **opts, extract_opts=extract_opts)
    assert tokens == ['väldig@JJ', 'romansk@JJ', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']


@pytest.mark.skip(reason="Not implemented")
def test_tagged_frame_to_token_counts():
    pass
    # def tagged_frame_to_token_counts(tagged_frame: TaggedFrame, pos_schema: PoS_Tag_Scheme, pos_column: str) -> dict:


def test_detect_phrases(tagged_frame: pd.DataFrame):

    found_phrases = detect_phrases(doc=tagged_frame, phrases=[], target="baseform")
    assert found_phrases == []

    found_phrases = detect_phrases(doc=tagged_frame, phrases=[["romansk"]], target="baseform")
    assert found_phrases == []

    found_phrases = detect_phrases(doc=tagged_frame, phrases=[["romansk", "kyrka"]], target="baseform")
    assert found_phrases == [(4, "romansk_kyrka", 2)]

    found_phrases = detect_phrases(
        doc=tagged_frame, phrases=[["väldig", "romansk"], ["romansk", "kyrka"]], target="baseform"
    )
    assert found_phrases == [(3, "väldig_romansk", 2), (4, "romansk_kyrka", 2)]


def test_merge_phrases_with_empty_list():
    tagged_frame: pd.DataFrame = create_tagged_frame()
    expected_tokens = tagged_frame.baseform.tolist()
    opts = dict(target_column="baseform", pad="*")  # pos_column="pos",
    tagged_frame = merge_phrases(doc=tagged_frame, phrase_positions=[], **opts)
    assert (tagged_frame.baseform == expected_tokens).all()


def test_merge_phrases_with_a_single_phrase():
    tagged_frame: pd.DataFrame = create_tagged_frame()
    opts = dict(target_column="baseform", pad="*")  # pos_column="pos",
    tagged_frame = merge_phrases(doc=tagged_frame, phrase_positions=[(4, "romansk_kyrka", 2)], **opts)
    assert (tagged_frame[3 : 6 + 1].baseform == ['väldig', 'romansk_kyrka', '*', 'tränga']).all()


def test_to_tagged_frame_with_phrase_detection():

    os.makedirs('./tests/output', exist_ok=True)
    opts = dict(
        extract_opts=ExtractTaggedTokensOpts(lemmatize=False),
        filter_opts=None,
        text_column='text',
        lemma_column='lemma_',
        pos_column='pos_',
        ignore_case=False,
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

    phrases = {'United_Nations': 'United Nations'.split(), 'United': ['United']}
    phrased_tokens = tagged_frame_to_tokens(tagged_frame, **opts, phrases=phrases)
    assert phrased_tokens[:9] == 'Constitution of the United_Nations Educational , Scientific and Cultural'.split(' ')

    # FIXME: #60 Regressed logic when prioritizing phrases (longer should apply=)
    # phrases = {'United_Nations': 'United Nations'.split(), 'the_United_Nations': 'the United Nations'.split()}
    # phrased_tokens = tagged_frame_to_tokens(tagged_frame, **opts, phrases=phrases)
    # assert phrased_tokens[:8] == 'Constitution of the_United_Nations Educational , Scientific and Cultural'.split(' ')

    phrases = {'United_Nations': 'united nations'.split()}

    phrased_tokens = tagged_frame_to_tokens(tagged_frame, phrases=phrases, **{**opts, **{'ignore_case': True}})
    assert phrased_tokens[:9] == 'constitution of the united_nations educational , scientific and cultural'.split(' ')
