import os
import shutil
import uuid

import penelope.co_occurrence as co_occurrence
import penelope.workflows.co_occurrence as workflow
import pytest
from penelope import corpus as corpora
from penelope import pipeline, utility
from penelope.pipeline.spacy.pipelines import spaCy_co_occurrence_pipeline
from penelope.workflows.interface import ComputeOpts

# pylint: disable=redefined-outer-name


def fake_config() -> pipeline.CorpusConfig:
    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load('./tests/test_data/SSI.yml')

    corpus_config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    corpus_config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'

    return corpus_config


@pytest.fixture(scope='module')
def config(en_nlp) -> pipeline.CorpusConfig:
    config: pipeline.CorpusConfig = fake_config()
    config.pipeline_payload.memory_store['spacy_model'] = en_nlp
    return config


@pytest.mark.long_running
def test_spaCy_co_occurrence_pipeline(config: pipeline.CorpusConfig):

    os.makedirs('./tests/output', exist_ok=True)
    tagged_corpus_source: str = "./tests/test_data/legal_instrument_five_docs_test_pos_csv.zip"
    target_filename = './tests/output/SSI-co-occurrence-JJVBNN-window-9.csv'
    if os.path.isfile(target_filename):
        os.remove(target_filename)

    # .folder(folder='./tests/test_data')
    pos_scheme: utility.PoS_Tag_Scheme = utility.PoS_Tag_Schemes.Universal
    transform_opts: corpora.TokensTransformOpts = corpora.TokensTransformOpts()
    extract_opts: corpora.ExtractTaggedTokensOpts = corpora.ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes=utility.pos_tags_to_str(pos_scheme.Adjective + pos_scheme.Verb + pos_scheme.Noun),
        pos_paddings=utility.pos_tags_to_str(pos_scheme.Conjunction),
        **config.pipeline_payload.tagged_columns_names,
        filter_opts=dict(is_punct=False),
    )
    context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(
        context_width=4,
        partition_keys=['document_id'],
    )
    global_threshold_count: int = 1

    value: co_occurrence.Bundle = spaCy_co_occurrence_pipeline(
        corpus_config=config,
        corpus_source=config.pipeline_payload.source,
        transform_opts=transform_opts,
        context_opts=context_opts,
        extract_opts=extract_opts,
        global_threshold_count=global_threshold_count,
        tagged_corpus_source=tagged_corpus_source,
    ).value()

    value.co_occurrences.to_csv(target_filename, sep='\t')

    assert os.path.isfile(target_filename)

    os.remove(target_filename)


@pytest.mark.long_running
@pytest.mark.skip(reason="Create fixture test case. Run manually!")
def test_spaCy_co_occurrence_workflow(config: pipeline.CorpusConfig):
    """Note: Use the output from this test case to update the tests/test_data/VENUS test data VENUS-TESTDATA"""

    os.makedirs('./tests/output', exist_ok=True)

    config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'
    config.checkpoint_opts.feather_folder = f'tests/output/{uuid.uuid1()}'
    corpus_tag: str = 'VENUS'
    target_folder: str = f'./tests/output/{uuid.uuid1()}'

    tagged_corpus_source: str = "./tests/output/co_occurrence_test_pos_csv.zip"

    bundle: co_occurrence.Bundle = spaCy_co_occurrence_pipeline(
        corpus_config=config,
        corpus_source=None,
        transform_opts=corpora.TokensTransformOpts(language='english', remove_stopwords=True, to_lower=True),
        extract_opts=corpora.ExtractTaggedTokensOpts(
            lemmatize=True,
            pos_includes='|NOUN|PROPN|VERB|',
            pos_excludes='|PUNCT|EOL|SPACE|',
            **config.pipeline_payload.tagged_columns_names,
            filter_opts=dict(is_alpha=False, is_punct=False, is_space=False),
        ),
        context_opts=co_occurrence.ContextOpts(
            context_width=4, ignore_concept=True, partition_keys=['document_id'], processes=None
        ),
        global_threshold_count=1,
        tagged_corpus_source=tagged_corpus_source,
    ).value()

    assert bundle.corpus is not None
    assert bundle.token2id is not None
    assert bundle.document_index is not None

    bundle.tag = corpus_tag
    bundle.folder = target_folder
    bundle.co_occurrences = bundle.corpus.to_co_occurrences(bundle.token2id)

    bundle.store()

    shutil.rmtree(bundle.folder, ignore_errors=True)
    shutil.rmtree(tagged_corpus_source, ignore_errors=True)
    shutil.rmtree(config.checkpoint_opts.feather_folder, ignore_errors=True)


@pytest.mark.long_running
def test_spaCy_co_occurrence_pipeline3(config):

    corpus_source = './tests/test_data/legal_instrument_five_docs_test.zip'
    tagged_corpus_source = f'./tests/output/{uuid.uuid1()}_pos.csv.zip'
    args: ComputeOpts = ComputeOpts(
        corpus_tag=f'{uuid.uuid1()}',
        corpus_source=corpus_source,
        target_folder=f'./tests/output/{uuid.uuid1()}',
        corpus_type=pipeline.CorpusType.SpacyCSV,
        # pos_scheme: utility.PoS_Tag_Scheme = utility.PoS_Tag_Schemes.Universal
        transform_opts=corpora.TokensTransformOpts(language='english', remove_stopwords=True, to_lower=True),
        text_reader_opts=corpora.TextReaderOpts(filename_pattern='*.csv', filename_fields=['year:_:1']),
        extract_opts=corpora.ExtractTaggedTokensOpts(
            lemmatize=True,
            pos_includes='|NOUN|PROPN|VERB|',
            pos_excludes='|PUNCT|EOL|SPACE|',
            **config.pipeline_payload.tagged_columns_names,
            filter_opts=dict(is_alpha=False, is_punct=False, is_space=False),
        ),
        create_subfolder=False,
        persist=True,
        vectorize_opts=corpora.VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            min_tf=1,
            max_tokens=None,
        ),
        enable_checkpoint=True,
        force_checkpoint=True,
        tf_threshold=1,
        tf_threshold_mask=False,
        context_opts=co_occurrence.ContextOpts(
            context_width=4,
            concept=set(),
            ignore_concept=False,
            partition_keys=['document_id'],
        ),
    )

    workflow.compute(
        args=args,
        corpus_config=config,
        tagged_corpus_source=tagged_corpus_source,
    )

    assert os.path.isfile(tagged_corpus_source)
    assert os.path.isdir(args.target_folder)

    shutil.rmtree(args.target_folder, ignore_errors=True)
    os.remove(tagged_corpus_source)
