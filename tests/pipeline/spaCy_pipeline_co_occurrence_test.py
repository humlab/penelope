import os

import pandas as pd
from penelope.co_occurrence import ContextOpts, store_bundle, to_vectorized_corpus
from penelope.corpus import TokensTransformOpts, VectorizedCorpus
from penelope.corpus.readers import ExtractTaggedTokensOpts, TaggedTokensFilterOpts
from penelope.notebook.co_occurrence.compute_callback_pipeline import compute_co_occurrence
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.spacy.pipelines import spaCy_co_occurrence_pipeline
from penelope.utility import pandas_to_csv_zip
from penelope.utility.pos_tags import PoS_Tag_Scheme, PoS_Tag_Schemes, pos_tags_to_str

from .fixtures import FakeGUI, FakeSSI


def test_spaCy_co_occurrence_pipeline():

    os.makedirs('./tests/output', exist_ok=True)
    checkpoint_filename: str = "./tests/test_data/SSI_tagged_frame_pos_csv.zip"
    target_filename = './tests/output/SSI-co-occurrence-JJVBNN-window-9.csv'
    if os.path.isfile(target_filename):
        os.remove(target_filename)

    ssi: CorpusConfig = CorpusConfig.load('./tests/test_data/ssi_corpus_config.yaml').files(
        source='./tests/test_data/legal_instrument_five_docs_test.zip',
        index_source='./tests/test_data/legal_instrument_five_docs_test.csv',
    )
    # .folder(folder='./tests/test_data')
    pos_scheme: PoS_Tag_Scheme = PoS_Tag_Schemes.Universal
    tokens_transform_opts: TokensTransformOpts = TokensTransformOpts()
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes=pos_tags_to_str(pos_scheme.Adjective + pos_scheme.Verb + pos_scheme.Noun)
    )
    tagged_tokens_filter_opts: TaggedTokensFilterOpts = TaggedTokensFilterOpts(
        is_space=False,
        is_punct=False,
    )
    context_opts: ContextOpts = ContextOpts(context_width=4)
    global_threshold_count: int = 1
    partition_column: str = 'year'

    co_occurrence = spaCy_co_occurrence_pipeline(
        corpus_config=ssi,
        tokens_transform_opts=tokens_transform_opts,
        extract_tagged_tokens_opts=extract_tagged_tokens_opts,
        tagged_tokens_filter_opts=tagged_tokens_filter_opts,
        context_opts=context_opts,
        global_threshold_count=global_threshold_count,
        partition_column=partition_column,
        checkpoint_filename=checkpoint_filename,
    ).value()

    co_occurrence.to_csv(target_filename, sep='\t')

    assert co_occurrence is not None
    assert os.path.isfile(target_filename)

    os.remove(target_filename)


def test_spaCy_co_occurrence_pipeline2():

    partition_key: str = 'year'
    corpus_config: CorpusConfig = FakeSSI(
        source='./tests/test_data/legal_instrument_five_docs_test.zip',
        index_source='./tests/test_data/legal_instrument_five_docs_test.csv',
        # source='./tests/test_data/legal_instrument_corpus.zip',
        # index_source='./tests/test_data/legal_instrument_index.csv',
    )
    args = FakeGUI(
        corpus_tag="VENUS",
        corpus_filename=corpus_config.pipeline_payload.source,
    )
    args.context_opts = ContextOpts(context_width=4, ignore_concept=True)  # , concept={''}, ignore_concept=True)

    os.makedirs('./tests/output', exist_ok=True)
    checkpoint_filename: str = "./tests/output/co_occurrence_test_pos_csv.zip"

    co_occurrences: pd.DataFrame = spaCy_co_occurrence_pipeline(
        corpus_config=corpus_config,
        tokens_transform_opts=args.tokens_transform_opts,
        extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
        tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
        context_opts=args.context_opts,
        global_threshold_count=args.count_threshold,
        partition_column=partition_key,
        checkpoint_filename=checkpoint_filename,
    ).value()

    assert co_occurrences is not None
    assert len(co_occurrences) > 0

    co_occurrence_filename = os.path.join(args.target_folder, f"{args.corpus_tag}_co-occurrence.csv.zip")

    pandas_to_csv_zip(
        zip_filename=co_occurrence_filename,
        dfs=(co_occurrences, 'co_occurrence.csv'),
        extension='csv',
        header=True,
        sep="\t",
        decimal=',',
        quotechar='"',
    )

    assert os.path.isfile(co_occurrence_filename)

    corpus: VectorizedCorpus = to_vectorized_corpus(co_occurrences=co_occurrences, value_column='value_n_t')

    store_bundle(
        co_occurrence_filename,
        corpus=corpus,
        corpus_tag=args.corpus_tag,
        input_filename=args.corpus_filename,
        partition_keys=[partition_key],
        count_threshold=args.count_threshold,
        co_occurrences=co_occurrences,
        reader_opts=corpus_config.text_reader_opts,
        tokens_transform_opts=args.tokens_transform_opts,
        context_opts=args.context_opts,
        extract_tokens_opts=args.extract_tagged_tokens_opts,
    )


def test_spaCy_co_occurrence_pipeline3():
    partition_key: str = 'year'
    corpus_config: CorpusConfig = FakeSSI(
        source='./tests/test_data/legal_instrument_five_docs_test.zip',
        index_source='./tests/test_data/legal_instrument_five_docs_test.csv',
        # source='./tests/test_data/legal_instrument_corpus.zip',
        # index_source='./tests/test_data/legal_instrument_index.csv',
    )
    args = FakeGUI(
        corpus_tag="VENUS",
        corpus_filename=corpus_config.pipeline_payload.source,
    )
    compute_co_occurrence(corpus_config=corpus_config, partition_key=partition_key, args=args, done_callback=None)
