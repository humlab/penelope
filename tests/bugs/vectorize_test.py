import contextlib
import uuid

import penelope.workflows.vectorize.dtm as workflow
import pytest
from penelope import corpus as corpora
from penelope import pipeline, utility
from penelope.notebook.interface import ComputeOpts

CORPUS_FOLDER = './tests/test_data'


@pytest.mark.long_running
def test_workflow_to_dtm():

    config: pipeline.CorpusConfig = pipeline.CorpusConfig.load('./tests/test_data/riksprot-kb-parlaclarin.yml')

    args: ComputeOpts = ComputeOpts(
        corpus_tag=f'{uuid.uuid1()}',
        corpus_source='/data/riksdagen_corpus_data/riksprot_parlaclarin_basic_protocol_stanza.csv.zip',
        corpus_type=pipeline.CorpusType.SparvCSV,
        target_folder='./data',
        transform_opts=corpora.TokensTransformOpts(to_lower=True, only_alphabetic=True),
        # text_reader_opts=corpora.TextReaderOpts(filename_pattern='*.csv', filename_fields=['year:_:1']),
        text_reader_opts=config.text_reader_opts,
        extract_opts=corpora.ExtractTaggedTokensOpts(
            lemmatize=True,
            pos_includes='',
            pos_excludes='|MID|MAD|PAD|',
            **config.pipeline_payload.tagged_columns_names,
        ),
        filter_opts=utility.PropertyValueMaskingOpts(),
        vectorize_opts=corpora.VectorizeOpts(already_tokenized=True, lowercase=False, verbose=False),
        create_subfolder=True,
        persist=True,
        enable_checkpoint=True,
        force_checkpoint=True,
        tf_threshold=5,
        tf_threshold_mask=True,
    )

    corpus = workflow.compute(args=args, corpus_config=config)

    corpus.remove(tag=args.corpus_tag, folder=args.target_folder)
    corpus.dump(tag=args.corpus_tag, folder=args.target_folder)

    assert corpora.VectorizedCorpus.dump_exists(tag=args.corpus_tag, folder=args.target_folder)

    corpus_loaded = corpora.VectorizedCorpus.load(tag=args.corpus_tag, folder=args.target_folder)

    assert corpus_loaded is not None

    y_corpus = corpus.group_by_year()

    assert y_corpus is not None

    with contextlib.suppress(Exception):
        corpus.remove(tag=args.corpus_tag, folder=args.target_folder)
