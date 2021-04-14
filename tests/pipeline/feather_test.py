import os
from typing import List
import pytest
import pandas as pd
from penelope.pipeline import CorpusConfig, CorpusPipeline, DocumentPayload
from penelope.utility import replace_extension

CORPUS_FOLDER = './tests/test_data'
OUTPUT_FOLDER = './tests/output'
# pylint: disable=redefined-outer-name


def fake_config() -> CorpusConfig:

    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/ssi_corpus_config.yml')

    corpus_config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    corpus_config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'

    return corpus_config


@pytest.fixture(scope='module')
def config():
    return fake_config()


def test_pipeline_can_can_be_saved_in_feather(config: CorpusConfig):

    checkpoint_filename: str = os.path.join(CORPUS_FOLDER, 'legal_instrument_five_docs_test_pos_csv.zip')

    pipeline = CorpusPipeline(config=config).checkpoint(checkpoint_filename)

    for payload in pipeline.resolve():
        
        tagged_frame: pd.DataFrame = payload.content
        filename = os.path.join(OUTPUT_FOLDER, replace_extension(payload.filename, ".feather"))
        if len(tagged_frame) == 0:
            continue

        tagged_frame.to_feather(filename, compression="lz4", )

        assert os.path.isfile(filename)

        apa = pd.read_feather(filename)

        assert apa is not None
