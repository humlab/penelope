import os
from io import StringIO
from typing import List

import numpy as np
import pandas as pd
import pytest

import penelope.utility.pos_tags as pos_tags
from penelope.corpus import ExtractTaggedTokensOpts, Token2Id, TokensTransformOpts
from penelope.corpus.readers import GLOBAL_TF_THRESHOLD_MASK_TOKEN
from penelope.pipeline import CheckpointOpts
from penelope.pipeline.convert import (
    PoSTagSchemaMissingError,
    Token2IdMissingError,
    filter_tagged_frame,
    tagged_frame_to_tokens,
)
from penelope.pipeline.phrases import detect_phrases, merge_phrases, parse_phrases
from penelope.pipeline.sparv import SparvCsvSerializer
from penelope.pipeline.sparv.convert import to_lemma_form
from penelope.type_alias import TaggedFrame
from tests.pipeline.fixtures import SPARV_TAGGED_COLUMNS

# pylint: disable=redefined-outer-name, no-member, unsubscriptable-object

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
fladdrade	VB	|fladdra omkring:10|
.	MAD	|
"""


def create_checkpoint_opts() -> CheckpointOpts:
    options: CheckpointOpts = CheckpointOpts(
        content_type_code=1,
        document_index_name=None,
        document_index_sep=None,
        sep='\t',
        quoting=3,
        custom_serializer_classname="penelope.pipeline.sparv.convert.SparvCsvSerializer",
        deserialize_processes=None,
        deserialize_chunksize=4,
        feather_folder=None,
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


def test_filter_tagged_frame(tagged_frame: pd.DataFrame):

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=False, pos_includes=None, pos_excludes=None, **SPARV_TAGGED_COLUMNS
    )
    filtered_frame = filter_tagged_frame(tagged_frame, extract_opts=extract_opts)
    assert filtered_frame.token.tolist() == tagged_frame.token.tolist()

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes=None, pos_excludes=None, **SPARV_TAGGED_COLUMNS)
    filtered_frame: pd.DataFrame = filter_tagged_frame(tagged_frame, extract_opts=extract_opts)
    assert (
        filtered_frame.token.tolist()
        == tagged_frame.apply(lambda x: to_lemma_form(x['token'], x['baseform']), axis=1).tolist()
    )

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=False, pos_includes='VB', pos_excludes=None, **SPARV_TAGGED_COLUMNS
    )
    filtered_frame: pd.DataFrame = filter_tagged_frame(tagged_frame, extract_opts=extract_opts)
    assert filtered_frame.token.tolist() == ['trängdes', 'gapade', 'fladdrade']

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes='VB', pos_excludes=None, **SPARV_TAGGED_COLUMNS)
    filtered_frame: pd.DataFrame = filter_tagged_frame(tagged_frame, extract_opts=extract_opts)
    assert filtered_frame.token.tolist() == ['tränga', 'gapa', 'fladdra_omkring']

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|VB|', pos_excludes=None, **SPARV_TAGGED_COLUMNS
    )
    filtered_frame: pd.DataFrame = filter_tagged_frame(tagged_frame, extract_opts=extract_opts)
    assert filtered_frame.token.tolist() == tagged_frame[tagged_frame.pos.isin(['VB'])].baseform.tolist()

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|VB|NN|', pos_excludes=None, **SPARV_TAGGED_COLUMNS
    )
    filtered_frame: pd.DataFrame = filter_tagged_frame(tagged_frame, extract_opts=extract_opts)
    assert filtered_frame.token.tolist() == tagged_frame[tagged_frame.pos.isin(['VB', 'NN'])].baseform.tolist()

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes=None, pos_excludes='MID|MAD|PAD', **SPARV_TAGGED_COLUMNS
    )
    filtered_frame: pd.DataFrame = filter_tagged_frame(tagged_frame, extract_opts=extract_opts)
    assert filtered_frame.token.tolist() == tagged_frame[~tagged_frame.pos.isin(['MID', 'MAD'])].baseform.tolist()

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|VB|', pos_excludes=None, **SPARV_TAGGED_COLUMNS
    )
    filtered_frame: pd.DataFrame = filter_tagged_frame(tagged_frame, extract_opts=extract_opts)
    assert filtered_frame.token.tolist() == tagged_frame[tagged_frame.pos.isin(['VB'])].baseform.tolist()


def create_encoded_tagged_frame(
    tagged_frame: pd.DataFrame,
    token2id: Token2Id,
    pos_schema: pos_tags.PoS_Tag_Scheme,
    target_column: str,
    pos_column: str,
) -> pd.DataFrame:

    id_tagged_frame: pd.DataFrame = pd.DataFrame(
        data=dict(
            token_id=tagged_frame[target_column].map(token2id).astype(np.int32),
            pos_id=tagged_frame[pos_column].map(pos_schema.pos_to_id).astype(np.int8),
        )
    )

    return id_tagged_frame


def test_encoded_filter_tagged_frame(tagged_frame: pd.DataFrame):

    pos_schema: pos_tags.PoS_Tag_Scheme = pos_tags.PoS_Tag_Schemes.SUC
    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts().set_numeric_names()
    """TEXT (NON-LEMMATIZED)"""
    token2id: Token2Id = Token2Id().ingest(tagged_frame['token'])
    tagged_id_frame: pd.DataFrame = create_encoded_tagged_frame(
        tagged_frame=tagged_frame,
        token2id=token2id,
        pos_schema=pos_schema,
        pos_column='pos',
        target_column='token',
    )
    with pytest.raises(Token2IdMissingError):
        _ = filter_tagged_frame(
            tagged_id_frame,
            extract_opts=extract_opts,
            token2id=None,
            pos_schema=pos_schema,
        )

    with pytest.raises(PoSTagSchemaMissingError):
        _ = filter_tagged_frame(
            tagged_id_frame,
            extract_opts=extract_opts,
            token2id=token2id,
            pos_schema=None,
        )

    filtered_frame = filter_tagged_frame(
        tagged_id_frame,
        extract_opts=extract_opts,
        token2id=token2id,
        pos_schema=pos_schema,
    )
    assert filtered_frame.token_id.tolist() == token2id.to_ids(tagged_frame.token.tolist())

    extract_opts = ExtractTaggedTokensOpts(pos_includes='VB').set_numeric_names()
    filtered_frame: pd.DataFrame = filter_tagged_frame(
        tagged_id_frame, token2id=token2id, pos_schema=pos_schema, extract_opts=extract_opts
    )
    assert filtered_frame.token_id.tolist() == token2id.to_ids(['trängdes', 'gapade', 'fladdrade'])

    """LEMMATIZED"""
    token2id: Token2Id = Token2Id().ingest(tagged_frame['baseform'])
    tagged_id_frame: pd.DataFrame = create_encoded_tagged_frame(
        tagged_frame=tagged_frame,
        token2id=token2id,
        pos_schema=pos_schema,
        pos_column='pos',
        target_column='baseform',
    )

    extract_opts = ExtractTaggedTokensOpts(pos_includes='VB').set_numeric_names()
    filtered_frame: pd.DataFrame = filter_tagged_frame(
        tagged_id_frame, token2id=token2id, pos_schema=pos_schema, extract_opts=extract_opts
    )
    assert filtered_frame.token_id.tolist() == token2id.to_ids(['tränga', 'gapa', 'fladdra_omkring'])

    extract_opts = ExtractTaggedTokensOpts(pos_includes='|VB|').set_numeric_names()
    filtered_frame: pd.DataFrame = filter_tagged_frame(
        tagged_id_frame, token2id=token2id, pos_schema=pos_schema, extract_opts=extract_opts
    )
    assert filtered_frame.token_id.tolist() == token2id.to_ids(
        tagged_frame[tagged_frame.pos.isin(['VB'])].baseform.tolist()
    )

    extract_opts = ExtractTaggedTokensOpts(pos_includes='|VB|NN|').set_numeric_names()
    filtered_frame: pd.DataFrame = filter_tagged_frame(
        tagged_id_frame, token2id=token2id, pos_schema=pos_schema, extract_opts=extract_opts
    )
    assert filtered_frame.token_id.tolist() == token2id.to_ids(
        tagged_frame[tagged_frame.pos.isin(['VB', 'NN'])].baseform.tolist()
    )

    extract_opts = ExtractTaggedTokensOpts(pos_includes=None, pos_excludes='MID|MAD|PAD').set_numeric_names()
    filtered_frame: pd.DataFrame = filter_tagged_frame(
        tagged_id_frame, token2id=token2id, pos_schema=pos_schema, extract_opts=extract_opts
    )
    assert filtered_frame.token_id.tolist() == token2id.to_ids(
        tagged_frame[~tagged_frame.pos.isin(['MID', 'MAD'])].baseform.tolist()
    )

    extract_opts = ExtractTaggedTokensOpts(pos_includes='|VB|').set_numeric_names()
    filtered_frame: pd.DataFrame = filter_tagged_frame(
        tagged_id_frame, token2id=token2id, pos_schema=pos_schema, extract_opts=extract_opts
    )
    assert filtered_frame.token_id.tolist() == token2id.to_ids(
        tagged_frame[tagged_frame.pos.isin(['VB'])].baseform.tolist()
    )


# @pytest.mark.parametrize("extract_opts,expected", [(dict(lemmatize=False, pos_includes=None, pos_excludes=None),[])])
def test_tagged_frame_to_tokens_pos_and_lemma(tagged_frame: pd.DataFrame):

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=False, pos_includes=None, pos_excludes=None, **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == tagged_frame.token.tolist()

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes=None, pos_excludes=None, **SPARV_TAGGED_COLUMNS)
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == tagged_frame.apply(lambda x: to_lemma_form(x['token'], x['baseform']), axis=1).tolist()

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=False, pos_includes='VB', pos_excludes=None, **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == ['trängdes', 'gapade', 'fladdrade']

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes='VB', pos_excludes=None, **SPARV_TAGGED_COLUMNS)
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == ['tränga', 'gapa', 'fladdra_omkring']

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|VB|', pos_excludes=None, **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == tagged_frame[tagged_frame.pos.isin(['VB'])].baseform.tolist()

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|VB|NN|', pos_excludes=None, **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == tagged_frame[tagged_frame.pos.isin(['VB', 'NN'])].baseform.tolist()

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes=None, pos_excludes='MID|MAD|PAD', **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == tagged_frame[~tagged_frame.pos.isin(['MID', 'MAD'])].baseform.tolist()

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|VB|', pos_excludes=None, **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == tagged_frame[tagged_frame.pos.isin(['VB'])].baseform.tolist()


def test_tagged_frame_to_tokens_with_passthrough(tagged_frame: pd.DataFrame):

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=False,
        pos_includes='VB',
        pos_excludes=None,
        passthrough_tokens=['kyrkan', 'ljuslågor'],
        **SPARV_TAGGED_COLUMNS,
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == ['kyrkan', 'trängdes', 'gapade', 'ljuslågor', 'fladdrade']


def test_tagged_frame_to_tokens_with_passthrough_and_blocks(tagged_frame: pd.DataFrame):

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=False,
        pos_includes='VB',
        pos_excludes=None,
        passthrough_tokens=['kyrkan', 'ljuslågor'],
        block_tokens=['fladdrade'],
        **SPARV_TAGGED_COLUMNS,
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == ['kyrkan', 'trängdes', 'gapade', 'ljuslågor']


def test_tagged_frame_to_tokens_with_global_tf_threshold(tagged_frame: pd.DataFrame):

    tagged_frame: pd.DataFrame = tagged_frame.copy()

    expected_counts: dict = {
        '.': 3,
        'bakom': 1,
        'den': 1,
        'fladdra_omkring': 1,
        'gapa': 1,
        'halvmörker': 1,
        'i': 2,
        'ingen': 1,
        'inne': 1,
        'kyrka': 1,
        'ljuslåga': 1,
        'någon': 1,
        'och': 1,
        'romansk': 1,
        'tränga': 1,
        'turist': 1,
        'valv': 2,
        'väldig': 1,
        'överblick': 1,
    }

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes=None, pos_excludes=None, **SPARV_TAGGED_COLUMNS)
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert set(expected_counts.keys()) == set(tokens)

    """TF threshold resets to 1 if token2id not supplied (i.e. token2id.TF is needed)"""
    extract_opts.global_tf_threshold = 2
    tokens = tagged_frame_to_tokens(tagged_frame, token2id=None, extract_opts=extract_opts)
    assert extract_opts.global_tf_threshold == 1

    token2id: Token2Id = Token2Id().ingest(["*", GLOBAL_TF_THRESHOLD_MASK_TOKEN]).ingest(tagged_frame.baseform)

    extract_opts.global_tf_threshold = 2
    extract_opts.global_tf_threshold_mask = False
    tokens = tagged_frame_to_tokens(tagged_frame, token2id=token2id, extract_opts=extract_opts)
    assert tokens == ['i', 'i', '.', 'valv', 'valv', '.', '.']

    extract_opts.global_tf_threshold = 2
    extract_opts.global_tf_threshold_mask = True
    tokens = tagged_frame_to_tokens(tagged_frame, token2id=token2id, extract_opts=extract_opts)
    assert len(tokens) == len(tagged_frame)
    assert set(tokens) == set([GLOBAL_TF_THRESHOLD_MASK_TOKEN, 'i', 'i', '.', 'valv', 'valv', '.', '.'])

    extract_opts.global_tf_threshold = 2
    extract_opts.global_tf_threshold_mask = True
    extract_opts.passthrough_tokens = {'överblick'}
    tokens = tagged_frame_to_tokens(tagged_frame, token2id=token2id, extract_opts=extract_opts)
    assert len(tokens) == len(tagged_frame)
    assert set(tokens) == set([GLOBAL_TF_THRESHOLD_MASK_TOKEN, 'i', 'i', '.', 'valv', 'valv', '.', '.', 'överblick'])


def test_tagged_frame_to_tokens_with_tf_threshold_and_threshold_tf_mask(tagged_frame: pd.DataFrame):

    extract_opts = ExtractTaggedTokensOpts(lemmatize=True, pos_includes=None, pos_excludes=None, **SPARV_TAGGED_COLUMNS)

    """ Alternative #1: tagged_frame_to_tokens does the filtering """

    df: pd.DataFrame = tagged_frame.copy()
    extract_opts.global_tf_threshold = 2
    extract_opts.global_tf_threshold_mask = True
    token2id: Token2Id = Token2Id().ingest(["*", GLOBAL_TF_THRESHOLD_MASK_TOKEN]).ingest(df.baseform)

    tokens = tagged_frame_to_tokens(df, token2id=token2id, extract_opts=extract_opts)
    assert len(tokens) == len(df)
    assert set(tokens) == set([GLOBAL_TF_THRESHOLD_MASK_TOKEN, 'i', 'i', '.', 'valv', 'valv', '.', '.'])

    """ Alternative #2: Use token2id to mask low TF tokens"""
    df: pd.DataFrame = tagged_frame.copy()
    token2id: Token2Id = Token2Id().ingest(["*", GLOBAL_TF_THRESHOLD_MASK_TOKEN]).ingest(df.baseform)
    """Note that translation must be used to map token-ids if used elsewhere"""
    _, translation = token2id.compress(tf_threshold=2, inplace=True)  # pylint: disable=unused-variable
    token2id.close()
    tokens = tagged_frame_to_tokens(df, token2id=token2id, extract_opts=extract_opts)
    assert len(tokens) == len(df)
    assert set(tokens) == set([GLOBAL_TF_THRESHOLD_MASK_TOKEN, 'i', 'i', '.', 'valv', 'valv', '.', '.'])


def test_tagged_frame_to_tokens_with_tf_threshold_and_not_threshold_tf_mask(tagged_frame: pd.DataFrame):

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes=None,
        pos_excludes=None,
        global_tf_threshold=2,
        global_tf_threshold_mask=False,
        **SPARV_TAGGED_COLUMNS,
    )
    """ Alternative #1: tagged_frame_to_tokens does the filtering """

    token2id: Token2Id = Token2Id().ingest(["*", GLOBAL_TF_THRESHOLD_MASK_TOKEN]).ingest(tagged_frame.baseform)
    expected_count = len(
        tagged_frame[
            tagged_frame.baseform.apply(lambda x: token2id.tf[token2id[x]] >= extract_opts.global_tf_threshold)
        ]
    )

    df: pd.DataFrame = tagged_frame.copy()
    tokens = tagged_frame_to_tokens(df, token2id=token2id, extract_opts=extract_opts)
    assert len(tokens) == expected_count
    assert set(tokens) == set(['i', 'i', '.', 'valv', 'valv', '.', '.'])

    """ Alternative #2: Use token2id to mask low TF tokens"""
    df: pd.DataFrame = tagged_frame.copy()
    token2id: Token2Id = Token2Id().ingest(["*", GLOBAL_TF_THRESHOLD_MASK_TOKEN]).ingest(df.baseform)
    """Note that translation must be used to map token-ids if used elsewhere"""
    _, translation = token2id.compress(tf_threshold=2, inplace=True)  # pylint: disable=unused-variable
    token2id.close()
    tokens = tagged_frame_to_tokens(df, token2id=token2id, extract_opts=extract_opts)
    assert len(tokens) == expected_count
    assert set(tokens) == set(['i', 'i', '.', 'valv', 'valv', '.', '.'])


def test_tagged_frame_to_tokens_replace_pos(tagged_frame: pd.DataFrame):

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes="NN", pos_excludes='MID|MAD|PAD', pos_paddings="VB", **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == ["kyrka", "*", "turist", "halvmörker", "valv", "*", "valv", "överblick", "ljuslåga", "*"]

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes=None, pos_excludes='MID|MAD|PAD', pos_paddings="VB|NN", **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == (
        ['inne', 'i', 'den', 'väldig', 'romansk', '*', '*', '*', 'i', '*', '*', '*']
        + ['bakom', '*', 'och', 'ingen', '*', 'någon', '*', '*']
    )

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes="JJ", pos_excludes='MID|MAD|PAD', pos_paddings="VB|NN", **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == ['väldig', 'romansk', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']


def test_tagged_frame_to_tokens_detect_phrases(tagged_frame: pd.DataFrame):

    expected_tokens = tagged_frame.baseform[:4].tolist() + ['romansk_kyrka'] + tagged_frame.baseform[6:].tolist()
    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes=None, phrases=[["romansk", "kyrka"]], **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == expected_tokens


def test_tagged_frame_to_tokens_with_append_pos_true(tagged_frame: pd.DataFrame):

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=False, pos_includes='VB', pos_excludes=None, append_pos=True, **SPARV_TAGGED_COLUMNS
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == ['trängdes@VB', 'gapade@VB', 'fladdrade@VB']

    extract_opts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes="JJ",
        pos_excludes='MID|MAD|PAD',
        pos_paddings="VB|NN",
        append_pos=True,
        **SPARV_TAGGED_COLUMNS,
    )
    tokens = tagged_frame_to_tokens(tagged_frame, extract_opts=extract_opts)
    assert tokens == ['väldig@JJ', 'romansk@JJ', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']


def test_detect_phrases(tagged_frame: pd.DataFrame):

    found_phrases = detect_phrases(tagged_frame["baseform"], phrases=[], ignore_case=True)
    assert not found_phrases

    found_phrases = detect_phrases(tagged_frame["baseform"], phrases=[["romansk"]], ignore_case=True)
    assert not found_phrases

    found_phrases = detect_phrases(tagged_frame["baseform"], phrases=[["romansk", "kyrka"]], ignore_case=True)
    assert found_phrases == [(4, "romansk_kyrka", 2)]

    found_phrases = detect_phrases(
        tagged_frame["baseform"], phrases=[["väldig", "romansk"], ["romansk", "kyrka"]], ignore_case=True
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


def test_parse_phrases():

    phrases_in_file_format: str = (
        "trazan_apansson; Trazan Apansson\nvery_good_gummisnodd;Very Good Gummisnodd\nkalle kula;Kalle Kula"
    )
    phrases_adhocs: List[str] = ["James Bond"]
    phrase_specification = parse_phrases(phrases_in_file_format, phrases_adhocs)

    assert phrase_specification == {
        "trazan_apansson": ["Trazan", "Apansson"],
        "very_good_gummisnodd": ["Very", "Good", "Gummisnodd"],
        "James_Bond": ["James", "Bond"],
        "kalle_kula": ["Kalle", "Kula"],
    }


def test_to_tagged_frame_SUC_pos_with_phrase_detection():

    os.makedirs('./tests/output', exist_ok=True)
    data_str: str = """token	pos	baseform
Herr	NN	|herr|
talman	NN	|talman|
!	MAD	|
Jag	PN	|jag|
ber	VB	|be|
få	VB	|få|
hemställa	VB	|hemställa|
,	MID	|
att	IE	|att|
kammaren	NN	|kammare|
måtte	VB	|må|
besluta	VB	|besluta|
att	IE	|att|
välja	VB	|välja|
suppleanter	NN	|suppleant|
i	PL	|
de	PN	|de|
ständiga	JJ	|ständig|
utskotten	NN	|utskott|
.	MAD	|
"""

    tagged_frame: pd.DataFrame = pd.read_csv(StringIO(data_str), sep='\t', index_col=None)

    phrases = {'herr_talman': 'Herr talman'.split()}
    phrased_tokens = tagged_frame_to_tokens(
        tagged_frame,
        extract_opts=ExtractTaggedTokensOpts(lemmatize=False, phrases=phrases, **SPARV_TAGGED_COLUMNS),
    )
    assert phrased_tokens[:9] == ['herr_talman', '!', 'Jag', 'ber', 'få', 'hemställa', ',', 'att', 'kammaren']


def transform_frame(tagged_frame: str, transform_opts: TokensTransformOpts) -> List[str]:

    tokens = tagged_frame_to_tokens(
        tagged_frame,
        extract_opts=ExtractTaggedTokensOpts(lemmatize=False, **SPARV_TAGGED_COLUMNS),
        transform_opts=transform_opts,
    )
    return tokens


def test_to_tagged_frame_to_tokens_with_transform_opts():

    os.makedirs('./tests/output', exist_ok=True)
    data_str: str = """token	pos	baseform
Herr	NN	|herr|
talman	NN	|talman|
!	MAD	|
Kammaren	NN	|kammare|
måste	VB	|må|
besluta	VB	|besluta|
att	IE	|att|
välja	VB	|välja|
10	RO	|10|
suppleanter	NN	|suppleant|
i	PL	|
utskotten	NN	|utskott|
.	MAD	|
"""
    tagged_frame: pd.DataFrame = pd.read_csv(StringIO(data_str), sep='\t', index_col=None)

    transform_opts: TokensTransformOpts() = None
    tokens = transform_frame(tagged_frame, transform_opts)
    assert tokens == tagged_frame.token.tolist()

    transform_opts: TokensTransformOpts = TokensTransformOpts(to_lower=True)
    tokens = transform_frame(tagged_frame, transform_opts)
    assert tokens == [x.lower() for x in tagged_frame.token.tolist()]

    # transform_opts: TokensTransformOpts = TokensTransformOpts(to_upper=True)
    # tokens = transform_frame(tagged_frame, transform_opts)
    # assert tokens == [ x.upper() for x in tagged_frame.token.tolist()]

    transform_opts: TokensTransformOpts = TokensTransformOpts(to_lower=True, only_alphabetic=True)
    tokens = transform_frame(tagged_frame, transform_opts)
    assert tokens == ['herr', 'talman', 'kammaren', 'måste', 'besluta', 'att', 'välja', 'suppleanter', 'i', 'utskotten']

    transform_opts: TokensTransformOpts = TokensTransformOpts(to_lower=True, only_any_alphanumeric=True)
    tokens = transform_frame(tagged_frame, transform_opts)
    assert tokens == [
        'herr',
        'talman',
        'kammaren',
        'måste',
        'besluta',
        'att',
        'välja',
        '10',
        'suppleanter',
        'i',
        'utskotten',
    ]

    transform_opts: TokensTransformOpts = TokensTransformOpts(to_lower=True, remove_stopwords=True, stopwords='swedish')
    tokens = transform_frame(tagged_frame, transform_opts)
    assert tokens == ['herr', 'talman', '!', 'kammaren', 'besluta', 'välja', '10', 'suppleanter', 'utskotten', '.']

    transform_opts: TokensTransformOpts = TokensTransformOpts(to_lower=True, min_len=5)
    tokens = transform_frame(tagged_frame, transform_opts)
    assert tokens == [x.lower() for x in tagged_frame.token.tolist() if len(x) >= 5]

    transform_opts: TokensTransformOpts = TokensTransformOpts(to_lower=True, max_len=5)
    tokens = transform_frame(tagged_frame, transform_opts)
    assert tokens == [x.lower() for x in tagged_frame.token.tolist() if len(x) <= 5]

    transform_opts: TokensTransformOpts = TokensTransformOpts(to_lower=True, keep_numerals=False, keep_symbols=False)
    tokens = transform_frame(tagged_frame, transform_opts)
    assert tokens == ['herr', 'talman', 'kammaren', 'måste', 'besluta', 'att', 'välja', 'suppleanter', 'i', 'utskotten']


def test_tagged_frame_to_token_counts(tagged_frame: TaggedFrame):

    pos_schema: pos_tags.PoS_Tag_Scheme = pos_tags.PoS_Tag_Schemes.SUC
    pos_column: str = "pos"

    group_counts = pos_schema.PoS_group_counts(PoS_sequence=tagged_frame[pos_column])
    assert group_counts == {
        'Adjective': 2,
        'Adverb': 2,
        'Conjunction': 0,
        'Delimiter': 3,
        'Noun': 7,
        'Numeral': 1,
        'Other': 1,
        'Preposition': 1,
        'Pronoun': 3,
        'Verb': 3,
        'n_raw_tokens': 20,
        'n_tokens': 20,
    }
