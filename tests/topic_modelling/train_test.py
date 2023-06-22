# flake8: noqa

import functools
import io
import os
import shutil
import uuid
from typing import Tuple

import pandas as pd
import pytest

from penelope import corpus as pc
from penelope import topic_modelling as tm
from penelope.topic_modelling.engines import get_engine_by_model_type
from penelope.topic_modelling.engines.engine_gensim import SUPPORTED_ENGINES, convert
from penelope.topic_modelling.engines.engine_gensim.utility import diagnostics_to_topic_token_weights_data
from penelope.topic_modelling.engines.interface import ITopicModelEngine
from penelope.vendor.gensim_api._gensim.wrappers.mallet_tm import MalletTopicModel
from tests.fixtures import TranströmerCorpus  # pylint: disable=non-ascii-module-import
from tests.utils import OUTPUT_FOLDER

from ..utils import PERSISTED_INFERRED_MODEL_SOURCE_FOLDER

# pylint: disable=ungrouped-imports,non-ascii-name,protected-access,redefined-outer-name


try:
    from penelope.scripts.tm.train_legacy import main
except (ImportError, NameError):
    ...


jj = os.path.join


def create_train_corpus() -> tm.TrainingCorpus:
    corpus: TranströmerCorpus = TranströmerCorpus()
    sparse_corpus, vocabulary = convert.TranslateCorpus().translate(corpus, id2token=None)
    tc: tm.TrainingCorpus = tm.TrainingCorpus(
        corpus=sparse_corpus,
        document_index=corpus.document_index,
        token2id=pc.Token2Id(vocabulary.token2id),
    )
    return tc


def _create_inferred_model(method: str, train_corpus: tm.TrainingCorpus) -> tm.InferredModel:
    inferred_model: tm.InferredModel = tm.train_model(
        train_corpus=train_corpus,
        method=method,
        engine_args={
            'n_topics': 4,
            'passes': 1,
            'random_seed': 42,
            'workers': 1,
            'max_iter': 100,
            'work_folder': f'./tests/output/{uuid.uuid4()}',
        },
    )

    return inferred_model


@functools.lru_cache(maxsize=None)
def create_model_data(method) -> Tuple[tm.TrainingCorpus, tm.InferredModel]:
    train_corpus: tm.TrainingCorpus = create_train_corpus()
    inferred_model: tm.InferredModel = _create_inferred_model(method, train_corpus)
    return train_corpus, inferred_model


def test_tranströmers_corpus():
    corpus = TranströmerCorpus()
    for filename, tokens in corpus:
        assert len(filename) > 0
        assert len(tokens) > 0


def test_create_train_corpus():
    pytest.importorskip("gensim")
    train_corpus: tm.TrainingCorpus = create_train_corpus()
    assert isinstance(train_corpus.document_index, pd.DataFrame)
    assert len(train_corpus.corpus) == len(train_corpus.document_index)
    assert len(train_corpus.document_index) == 5
    assert len(train_corpus.document_index.columns) == 8
    assert 'n_tokens' in train_corpus.document_index.columns


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_infer_model(method):
    pytest.importorskip("gensim")

    _, inferred_model = create_model_data(method)

    assert inferred_model is not None
    assert inferred_model.method == method

    engine: ITopicModelEngine = get_engine_by_model_type(inferred_model.topic_model)
    assert isinstance(inferred_model.topic_model, engine.supported_models())


def test_load_inferred_model_fixture():
    inferred_model: tm.InferredModel = tm.InferredModel.load(PERSISTED_INFERRED_MODEL_SOURCE_FOLDER)
    assert inferred_model is not None


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_store_compressed_inferred_model(method):
    pytest.importorskip("gensim")

    _, inferred_model = create_model_data(method)
    target_name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, target_name)

    inferred_model.store(target_folder, store_compressed=False)

    assert os.path.isfile(os.path.join(target_folder, "model_options.json"))

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_store_uncompressed_inferred_model(method):
    pytest.importorskip("gensim")

    train_corpus, inferred_model = create_model_data(method)
    target_name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, target_name)

    inferred_model.store(target_folder, store_compressed=True)
    train_corpus.store(target_folder)

    assert pc.VectorizedCorpus.dump_exists(folder=target_folder, tag="train")
    assert os.path.isfile(os.path.join(target_folder, "model_options.json"))

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("engine_key", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_load_inferred_model(engine_key):
    pytest.importorskip("gensim")

    target_name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, target_name)
    _, test_inferred_model = create_model_data(engine_key)
    test_inferred_model.store(target_folder, store_compressed=True)

    """Corpus no longer stored by default"""
    train_corpus = tm.TrainingCorpus.load(target_folder)
    assert train_corpus is None

    inferred_model: tm.InferredModel = tm.InferredModel.load(target_folder)

    assert inferred_model is not None
    assert inferred_model.method == engine_key
    assert isinstance(inferred_model.topic_model, SUPPORTED_ENGINES[engine_key].engine)
    shutil.rmtree(target_folder)


def test_load_trained_corpus():
    pytest.importorskip("gensim")

    target_name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, target_name)

    train_corpus = create_train_corpus()
    train_corpus.store(target_folder)
    assert train_corpus.exists(target_folder)

    train_corpus = tm.TrainingCorpus.load(target_folder)

    assert isinstance(train_corpus.document_index, pd.DataFrame)
    assert len(train_corpus.document_index) == 5
    assert len(train_corpus.document_index.columns) == 8
    assert 'n_tokens' in train_corpus.document_index.columns
    assert train_corpus.corpus is not None

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("engine_key", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_load_inferred_model_when_stored_corpus_is_false_has_no_trained_corpus(engine_key):
    pytest.importorskip("gensim")

    target_name: str = f"{uuid.uuid1()}"
    target_folder: str = os.path.join(OUTPUT_FOLDER, target_name)
    _, test_inferred_model = create_model_data(engine_key)
    test_inferred_model.store(target_folder)

    inferred_model = tm.InferredModel.load(target_folder)

    assert inferred_model is not None
    assert inferred_model.method == engine_key
    assert isinstance(inferred_model.topic_model, SUPPORTED_ENGINES[engine_key].engine)

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_load_inferred_model_when_lazy_does_not_load_model_or_corpus(method):
    pytest.importorskip("gensim")

    target_name = f"{uuid.uuid1()}"
    target_folder = jj(OUTPUT_FOLDER, target_name)
    _, test_inferred_model = create_model_data(method)
    test_inferred_model.store(target_folder)

    inferred_model: tm.InferredModel = tm.InferredModel.load(target_folder, lazy=True)

    assert callable(inferred_model._topic_model)

    _ = inferred_model.topic_model

    assert not callable(inferred_model._topic_model)

    _ = inferred_model.topic_model

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_infer_topics_data(method):
    pytest.importorskip("gensim")

    minimum_probability: float = 0.001
    n_tokens: int = 5

    train_corpus, inferred_model = create_model_data(method)

    inferred_topics_data: tm.InferredTopicsData = tm.predict_topics(
        topic_model=inferred_model.topic_model,
        corpus=train_corpus.corpus,
        id2token=train_corpus.id2token,
        document_index=train_corpus.document_index,
        minimum_probability=minimum_probability,
        n_tokens=n_tokens,
    )

    assert inferred_topics_data is not None
    assert isinstance(inferred_topics_data.document_index, pd.DataFrame)
    assert isinstance(inferred_topics_data.dictionary, pd.DataFrame)
    assert isinstance(inferred_topics_data.topic_token_weights, pd.DataFrame)
    assert isinstance(inferred_topics_data.topic_token_overview, pd.DataFrame)
    assert isinstance(inferred_topics_data.document_topic_weights, pd.DataFrame)
    assert inferred_topics_data.year_period == (2019, 2020)
    assert set(inferred_topics_data.topic_ids) == {0, 1, 2, 3}
    assert len(inferred_topics_data.document_index) == 5
    assert list(inferred_topics_data.topic_token_weights.topic_id.unique()) == [0, 1, 2, 3]
    assert list(inferred_topics_data.topic_token_overview.index) == [0, 1, 2, 3]
    assert set(inferred_topics_data.document_topic_weights.topic_id.unique()) == {0, 1, 2, 3}


@pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_run_cli(method):
    pytest.importorskip("gensim")

    kwargs = {
        'target_name': f"{uuid.uuid1()}",
        'corpus_folder': OUTPUT_FOLDER,
        'corpus_source': './tests/test_data/test_corpus.zip',
        'engine': method,
        'engine_args': {
            'n_topics': 5,
            'alpha': 'asymmetric',
        },
        'filename_field': ('year:_:1', 'sequence_id:_:2'),
    }

    main(**kwargs)

    target_folder = jj(kwargs['corpus_folder'], kwargs['target_name'])

    assert os.path.isdir(target_folder)
    assert os.path.isfile(jj(target_folder, 'topic_model.pickle.pbz2'))
    assert os.path.isfile(jj(target_folder, 'model_options.json'))

    inferred_model: tm.InferredModel = tm.InferredModel.load(target_folder)
    inferred_topic_data: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=target_folder)

    assert inferred_model is not None
    assert inferred_topic_data is not None

    shutil.rmtree(target_folder)


DIAGNOSTICS_XML: str = """<?xml version="1.0" encoding="UTF-8"?>
<model>
<topic id='0' tokens='31.0000' document_entropy='1.3643' word-length='6.5000' coherence='-139.3440' uniform_dist='1.6239' corpus_dist='1.4102' eff_num_words='18.1321' token-doc-diff='0.0462' rank_1_docs='0.4000' allocation_ratio='0.0000' allocation_count='0.0000' exclusivity='0.9754'>
<word rank='1' count='5' prob='0.16129' cumulative='0.16129' docs='1' word-length='4.0000' coherence='0.0000' uniform_dist='0.4724' corpus_dist='0.2275' token-doc-diff='0.0368' exclusivity='0.9942'>valv</word>
<word rank='2' count='2' prob='0.06452' cumulative='0.22581' docs='2' word-length='6.0000' coherence='-4.6151' uniform_dist='0.1299' corpus_dist='0.0910' token-doc-diff='0.0019' exclusivity='0.9856'>träden</word>
<word rank='3' count='1' prob='0.03226' cumulative='0.25806' docs='1' word-length='4.0000' coherence='-4.6151' uniform_dist='0.0426' corpus_dist='0.0455' token-doc-diff='0.0009' exclusivity='0.9718'>kust</word>
<word rank='4' count='1' prob='0.03226' cumulative='0.29032' docs='1' word-length='8.0000' coherence='-4.6151' uniform_dist='0.0426' corpus_dist='0.0455' token-doc-diff='0.0009' exclusivity='0.9718'>sommarns</word>
<word rank='5' count='1' prob='0.03226' cumulative='0.32258' docs='1' word-length='6.0000' coherence='-4.6151' uniform_dist='0.0426' corpus_dist='0.0455' token-doc-diff='0.0009' exclusivity='0.9718'>frihet</word>
<word rank='6' count='1' prob='0.03226' cumulative='0.35484' docs='1' word-length='4.0000' coherence='-4.6151' uniform_dist='0.0426' corpus_dist='0.0455' token-doc-diff='0.0009' exclusivity='0.9718'>året</word>
<word rank='7' count='1' prob='0.03226' cumulative='0.38710' docs='1' word-length='13.0000' coherence='-5.3033' uniform_dist='0.0426' corpus_dist='0.0455' token-doc-diff='0.0009' exclusivity='0.9718'>fladdermössen</word>
<word rank='8' count='1' prob='0.03226' cumulative='0.41935' docs='1' word-length='9.0000' coherence='-4.6151' uniform_dist='0.0426' corpus_dist='0.0455' token-doc-diff='0.0009' exclusivity='0.9718'>sköldarna</word>
<word rank='9' count='1' prob='0.03226' cumulative='0.45161' docs='1' word-length='8.0000' coherence='-4.6151' uniform_dist='0.0426' corpus_dist='0.0455' token-doc-diff='0.0009' exclusivity='0.9718'>gångstig</word>
<word rank='10' count='1' prob='0.03226' cumulative='0.48387' docs='1' word-length='3.0000' coherence='-4.6151' uniform_dist='0.0426' corpus_dist='0.0455' token-doc-diff='0.0009' exclusivity='0.9718'>öde</word>
</topic>
<topic id='1' tokens='30.0000' document_entropy='1.3649' word-length='8.9000' coherence='-121.5717' uniform_dist='1.3986' corpus_dist='1.4430' eff_num_words='28.1250' token-doc-diff='0.0000' rank_1_docs='0.0000' allocation_ratio='0.0000' allocation_count='0.0000' exclusivity='0.9743'>
<word rank='1' count='2' prob='0.06667' cumulative='0.06667' docs='2' word-length='3.0000' coherence='0.0000' uniform_dist='0.1364' corpus_dist='0.0962' token-doc-diff='0.0000' exclusivity='0.9862'>men</word>
<word rank='2' count='1' prob='0.03333' cumulative='0.10000' docs='1' word-length='19.0000' coherence='-0.6882' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0000' exclusivity='0.9730'>barrskogsbränningen</word>
<word rank='3' count='1' prob='0.03333' cumulative='0.13333' docs='1' word-length='7.0000' coherence='-0.6882' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0000' exclusivity='0.9730'>bergets</word>
<word rank='4' count='1' prob='0.03333' cumulative='0.16667' docs='1' word-length='4.0000' coherence='-0.6882' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0000' exclusivity='0.9730'>vind</word>
<word rank='5' count='1' prob='0.03333' cumulative='0.20000' docs='1' word-length='9.0000' coherence='-0.6882' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0000' exclusivity='0.9730'>stövlarna</word>
<word rank='6' count='1' prob='0.03333' cumulative='0.23333' docs='1' word-length='5.0000' coherence='-5.3033' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0000' exclusivity='0.9730'>havet</word>
<word rank='7' count='1' prob='0.03333' cumulative='0.26667' docs='1' word-length='12.0000' coherence='-4.6151' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0000' exclusivity='0.9730'>flygvingarna</word>
<word rank='8' count='1' prob='0.03333' cumulative='0.30000' docs='1' word-length='19.0000' coherence='-4.6151' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0000' exclusivity='0.9730'>kommunikationsnätet</word>
<word rank='9' count='1' prob='0.03333' cumulative='0.33333' docs='1' word-length='6.0000' coherence='-4.6151' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0000' exclusivity='0.9730'>Stegen</word>
<word rank='10' count='1' prob='0.03333' cumulative='0.36667' docs='1' word-length='5.0000' coherence='-4.6151' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0000' exclusivity='0.9730'>sidan</word>
</topic>
<topic id='2' tokens='30.0000' document_entropy='1.4067' word-length='7.3000' coherence='-143.2709' uniform_dist='1.4910' corpus_dist='1.4430' eff_num_words='25.0000' token-doc-diff='0.0108' rank_1_docs='0.2000' allocation_ratio='0.0000' allocation_count='0.0000' exclusivity='0.9769'>
<word rank='1' count='2' prob='0.06667' cumulative='0.06667' docs='1' word-length='13.0000' coherence='0.0000' uniform_dist='0.1364' corpus_dist='0.0962' token-doc-diff='0.0041' exclusivity='0.9862'>grundstenarna</word>
<word rank='2' count='2' prob='0.06667' cumulative='0.13333' docs='1' word-length='6.0000' coherence='0.0000' uniform_dist='0.1364' corpus_dist='0.0962' token-doc-diff='0.0041' exclusivity='0.9862'>skugga</word>
<word rank='3' count='2' prob='0.06667' cumulative='0.20000' docs='2' word-length='5.0000' coherence='0.0000' uniform_dist='0.1364' corpus_dist='0.0962' token-doc-diff='0.0006' exclusivity='0.9862'>solen</word>
<word rank='4' count='1' prob='0.03333' cumulative='0.23333' docs='1' word-length='9.0000' coherence='-4.6151' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0003' exclusivity='0.9730'>ögonblick</word>
<word rank='5' count='1' prob='0.03333' cumulative='0.26667' docs='1' word-length='7.0000' coherence='-4.6151' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0003' exclusivity='0.9730'>trädens</word>
<word rank='6' count='1' prob='0.03333' cumulative='0.30000' docs='1' word-length='6.0000' coherence='-4.6151' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0003' exclusivity='0.9730'>dyning</word>
<word rank='7' count='1' prob='0.03333' cumulative='0.33333' docs='1' word-length='7.0000' coherence='-5.3033' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0003' exclusivity='0.9730'>stjärna</word>
<word rank='8' count='1' prob='0.03333' cumulative='0.36667' docs='1' word-length='6.0000' coherence='-5.3033' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0003' exclusivity='0.9730'>Vraken</word>
<word rank='9' count='1' prob='0.03333' cumulative='0.40000' docs='1' word-length='6.0000' coherence='-5.3033' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0003' exclusivity='0.9730'>Jorden</word>
<word rank='10' count='1' prob='0.03333' cumulative='0.43333' docs='1' word-length='8.0000' coherence='-5.3033' uniform_dist='0.0451' corpus_dist='0.0481' token-doc-diff='0.0003' exclusivity='0.9730'>stranden</word>
</topic>
<topic id='3' tokens='36.0000' document_entropy='1.4169' word-length='5.5000' coherence='-131.6923' uniform_dist='1.2471' corpus_dist='1.2607' eff_num_words='32.4000' token-doc-diff='0.0000' rank_1_docs='0.4000' allocation_ratio='0.0000' allocation_count='0.0000' exclusivity='0.9694'>
<word rank='1' count='2' prob='0.05556' cumulative='0.05556' docs='2' word-length='6.0000' coherence='0.0000' uniform_dist='0.1035' corpus_dist='0.0700' token-doc-diff='0.0000' exclusivity='0.9827'>ljuset</word>
<word rank='2' count='2' prob='0.05556' cumulative='0.11111' docs='2' word-length='2.0000' coherence='-0.6882' uniform_dist='0.1035' corpus_dist='0.0700' token-doc-diff='0.0000' exclusivity='0.9827'>nu</word>
<word rank='3' count='1' prob='0.02778' cumulative='0.13889' docs='1' word-length='6.0000' coherence='-5.3033' uniform_dist='0.0325' corpus_dist='0.0350' token-doc-diff='0.0000' exclusivity='0.9661'>toppar</word>
<word rank='4' count='1' prob='0.02778' cumulative='0.16667' docs='1' word-length='3.0000' coherence='-5.3033' uniform_dist='0.0325' corpus_dist='0.0350' token-doc-diff='0.0000' exclusivity='0.9661'>fot</word>
<word rank='5' count='1' prob='0.02778' cumulative='0.19444' docs='1' word-length='6.0000' coherence='-5.3033' uniform_dist='0.0325' corpus_dist='0.0350' token-doc-diff='0.0000' exclusivity='0.9661'>mörker</word>
<word rank='6' count='1' prob='0.02778' cumulative='0.22222' docs='1' word-length='6.0000' coherence='-5.3033' uniform_dist='0.0325' corpus_dist='0.0350' token-doc-diff='0.0000' exclusivity='0.9661'>betsel</word>
<word rank='7' count='1' prob='0.02778' cumulative='0.25000' docs='1' word-length='8.0000' coherence='-5.3033' uniform_dist='0.0325' corpus_dist='0.0350' token-doc-diff='0.0000' exclusivity='0.9661'>stillhet</word>
<word rank='8' count='1' prob='0.02778' cumulative='0.27778' docs='1' word-length='5.0000' coherence='-5.3033' uniform_dist='0.0325' corpus_dist='0.0350' token-doc-diff='0.0000' exclusivity='0.9661'>punkt</word>
<word rank='9' count='1' prob='0.02778' cumulative='0.30556' docs='1' word-length='7.0000' coherence='-5.3033' uniform_dist='0.0325' corpus_dist='0.0350' token-doc-diff='0.0000' exclusivity='0.9661'>vrakens</word>
<word rank='10' count='1' prob='0.02778' cumulative='0.33333' docs='1' word-length='6.0000' coherence='-4.6151' uniform_dist='0.0325' corpus_dist='0.0350' token-doc-diff='0.0000' exclusivity='0.9661'>expert</word>
</topic>
</model>
"""


def test_parse_diagnostics():
    topics: pd.DataFrame = pd.read_xml(DIAGNOSTICS_XML, xpath=".//topic")
    words: pd.DataFrame = pd.read_xml(DIAGNOSTICS_XML, xpath=".//word")

    # diags: untangle.Element = untangle.parse(DIAGNOSTICS_XML)
    # topics: pd.DataFrame = pd.DataFrame([t.attributes for t in diags.model.topic]).set_index('id')
    # words: pd.DataFrame = pd.DataFrame(
    #     [
    #         {
    #             **{'topic_id': t['id']},
    #             **w.attributes,
    #         }
    #         for t in diags.model.topic
    #         for w in t.word
    #     ]
    # )

    assert words is not None
    assert topics is not None


def test_sax_parse():
    x = [x for x in MalletTopicModel.parse_diagnostics_words(io.StringIO(DIAGNOSTICS_XML))]
    assert x
    assert x[0]['token'] == 'valv'
    assert x[0]['topic_id'] == 0

    assert x[-1]['token'] == 'expert'
    assert x[-1]['topic_id'] == 3


def test_diagnostics_to_topic_token_weights_data():
    topic_token_diagnostics = MalletTopicModel.load_topic_token_diagnostics2(io.StringIO(DIAGNOSTICS_XML))

    assert topic_token_diagnostics is not None
    data = diagnostics_to_topic_token_weights_data(topic_token_diagnostics, n_tokens=10)
    assert data is not None
    assert len(data) == 4
    assert [x[0] for x in data] == [0, 1, 2, 3]

    topic_id, token_weights = data[0]
    assert topic_id == 0
    assert (token_weights[0], token_weights[-1]) == (('valv', 0.16129), ('öde', 0.03226))

    topic_id, token_weights = data[3]
    assert topic_id == 3
    assert (token_weights[0], token_weights[-1]) == (('ljuset', 0.05556), ('expert', 0.02778))
