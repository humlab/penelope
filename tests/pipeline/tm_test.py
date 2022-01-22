import os
from os.path import join as jj

import pytest

from penelope import corpus as pc
from penelope import pipeline as pp
from penelope import topic_modelling as tm
from penelope import utility


def noun_pipeline(id_to_token: bool) -> pp.CorpusPipeline:
    corpus_source: str = './tests/test_data/tranströmer_id_tagged_frames'
    file_pattern: str = '**/tran_*.feather'

    config_filename: str = jj(corpus_source, 'corpus.yml')
    corpus_config: pp.CorpusConfig = pp.CorpusConfig.load(path=config_filename).folders(corpus_source)
    extract_opts: pc.ExtractTaggedTokensOpts = pc.ExtractTaggedTokensOpts(
        lemmatize=False, pos_includes='NN', **corpus_config.pipeline_payload.tagged_columns_names
    )

    if not id_to_token:
        extract_opts.set_numeric_names()

    p: pp.CorpusPipeline = (
        pp.CorpusPipeline(config=corpus_config)
        .load_id_tagged_frame(folder=corpus_source, id_to_token=id_to_token, file_pattern=file_pattern)
        .filter_tagged_frame(extract_opts=extract_opts, pos_schema=utility.PoS_Tag_Schemes.SUC)
    )
    return p


def noun_dtm_pipeline(min_tf: int = 1, max_tokens: int = None) -> pp.CorpusPipeline:

    vectorize_opts: pc.VectorizeOpts = pc.VectorizeOpts(
        already_tokenized=True, lowercase=False, min_tf=min_tf, max_tokens=max_tokens
    )

    p: pp.CorpusPipeline = noun_pipeline(id_to_token=False).to_dtm(
        vectorize_opts=vectorize_opts, tagged_column='token_id'
    )

    return p


def test_load_id_tagged_frame_pipeline():

    p: pp.CorpusPipeline = noun_pipeline(id_to_token=False)
    payloads = p.to_list()

    assert len(payloads) == 5
    assert 'pos_id' in payloads[0].content.columns
    assert 'token_id' in payloads[0].content.columns


def test_load_id_tagged_frame_pipeline_convert_ids_to_text():
    p: pp.CorpusPipeline = noun_pipeline(id_to_token=True)
    payloads = p.to_list()

    assert len(payloads) == 5
    assert 'pos' in payloads[0].content.columns
    assert 'token' in payloads[0].content.columns


def test_load_id_tagged_frame_pipeline_convert_to_dtm():

    corpus: pc.VectorizedCorpus = noun_dtm_pipeline(min_tf=1, max_tokens=None).value()
    assert corpus.shape == (5, 116)

    corpus: pc.VectorizedCorpus = noun_dtm_pipeline(min_tf=1, max_tokens=10).value()
    assert corpus.shape == (5, 10)


@pytest.mark.parametrize('method', ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_load_id_tagged_frame_pipeline_convert_to_topic_model(method):

    os.makedirs('./tests/output/tmp', exist_ok=True)

    target_name: str = 'KALLE'  # f'{str(uuid.uuid1())[:6]}'
    target_folder: str = "./tests/output"
    engine_args: dict = {
        'n_topics': 4,
        'passes': 1,
        'random_seed': 42,
        'workers': 1,
        'max_iter': 100,
        'work_folder': './tests/output/tmp',
    }

    """Train and predict from FEATHER source"""
    value: dict = (
        noun_dtm_pipeline(min_tf=1, max_tokens=None)
        .to_topic_model(
            target_mode='both',
            target_folder=target_folder,
            target_name=target_name,
            engine=method,
            engine_args=engine_args,
            store_corpus=True,
            store_compressed=True,
            train_corpus_folder=None,
            trained_model_folder=None,
            n_tokens=200,
            minimum_probability=0.01,
        )
        .value()
    )

    assert isinstance(value, dict) is not None
    assert value.get('target_name') == target_name
    assert value.get('target_folder') == target_folder
    assert tm.InferredModel.exists(jj(target_folder, target_name))

    """Train and predict using existing training corpus"""
    target_name2: str = 'KULA'  # f'{str(uuid.uuid1())[:6]}'
    value2: dict = (
        noun_dtm_pipeline(min_tf=1, max_tokens=None)
        .to_topic_model(
            target_mode='both',
            target_folder=target_folder,
            target_name=target_name2,
            engine=method,
            engine_args=engine_args,
            store_corpus=True,
            store_compressed=True,
            train_corpus_folder=jj(target_folder, target_name),
            trained_model_folder=None,
            n_tokens=200,
            minimum_probability=0.01,
        )
        .value()
    )

    assert isinstance(value2, dict) is not None
    assert value2.get('target_name') == target_name2
    assert value2.get('target_folder') == target_folder
    assert tm.InferredModel.exists(jj(value2.get('target_folder'), value2.get('target_name')))

    """Train and predict using existing corpus and model"""

    assert True

    target_name3: str = 'KURT'  # f'{str(uuid.uuid1())[:6]}'
    value3: dict = (
        noun_dtm_pipeline(min_tf=1, max_tokens=None)
        .to_topic_model(
            target_mode='predict',
            target_folder=target_folder,
            target_name=target_name3,
            engine=method,
            engine_args=engine_args,
            store_corpus=True,
            store_compressed=True,
            train_corpus_folder=None,
            trained_model_folder=jj(target_folder, target_name),
            n_tokens=200,
            minimum_probability=0.01,
        )
        .value()
    )

    assert isinstance(value3, dict) is not None
    assert value3.get('target_name') == target_name3
    assert value3.get('target_folder') == target_folder
    assert not tm.InferredModel.exists(jj(value3.get('target_folder'), value3.get('target_name')))
    assert os.path.isfile(jj(value3.get('target_folder'), value3.get('target_name'), 'document_topic_weights.zip'))
