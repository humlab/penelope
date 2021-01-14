import os

import penelope.co_occurrence as co_occurrence
from penelope.co_occurrence.partitioned import ComputeResult
from penelope.corpus import TokensTransformOpts, VectorizedCorpus
from penelope.corpus.readers import ExtractTaggedTokensOpts, TaggedTokensFilterOpts
from penelope.notebook.co_occurrence.compute_pipeline import compute_co_occurrence
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.spacy.pipelines import spaCy_co_occurrence_pipeline
from penelope.utility.pos_tags import PoS_Tag_Scheme, PoS_Tag_Schemes, pos_tags_to_str

from ..fixtures import FakeComputeOptsSpacyCSV, FakeSSI


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
        is_punct=False,
    )
    context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(context_width=4)
    global_threshold_count: int = 1
    partition_column: str = 'year'

    compute_result: ComputeResult = spaCy_co_occurrence_pipeline(
        corpus_config=ssi,
        tokens_transform_opts=tokens_transform_opts,
        extract_tagged_tokens_opts=extract_tagged_tokens_opts,
        tagged_tokens_filter_opts=tagged_tokens_filter_opts,
        context_opts=context_opts,
        global_threshold_count=global_threshold_count,
        partition_column=partition_column,
        checkpoint_filename=checkpoint_filename,
    ).value()

    compute_result.co_occurrences.to_csv(target_filename, sep='\t')

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
    args = FakeComputeOptsSpacyCSV(
        corpus_tag="VENUS",
        corpus_filename=corpus_config.pipeline_payload.source,
    )
    args.context_opts = co_occurrence.ContextOpts(
        context_width=4, ignore_concept=True
    )  # , concept={''}, ignore_concept=True)

    os.makedirs('./tests/output', exist_ok=True)
    checkpoint_filename: str = "./tests/output/co_occurrence_test_pos_csv.zip"

    compute_result: co_occurrence.ComputeResult = spaCy_co_occurrence_pipeline(
        corpus_config=corpus_config,
        tokens_transform_opts=args.tokens_transform_opts,
        extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
        tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
        context_opts=args.context_opts,
        global_threshold_count=args.count_threshold,
        partition_column=partition_key,
        checkpoint_filename=checkpoint_filename,
    ).value()

    assert compute_result.co_occurrences is not None
    assert compute_result.document_index is not None
    assert len(compute_result.co_occurrences) > 0

    co_occurrence_filename = co_occurrence.folder_and_tag_to_filename(folder=args.target_folder, tag=args.corpus_tag)

    corpus: VectorizedCorpus = co_occurrence.to_vectorized_corpus(
        co_occurrences=compute_result.co_occurrences, document_index=compute_result.document_index, value_column='value'
    )

    bundle = co_occurrence.Bundle(
        corpus=corpus,
        corpus_tag=args.corpus_tag,
        corpus_folder=args.target_folder,
        co_occurrences=compute_result.co_occurrences,
        document_index=compute_result.document_index,
        compute_options=co_occurrence.create_options_bundle(
            reader_opts=corpus_config.text_reader_opts,
            tokens_transform_opts=args.tokens_transform_opts,
            context_opts=args.context_opts,
            extract_tokens_opts=args.extract_tagged_tokens_opts,
            input_filename=args.corpus_filename,
            output_filename=co_occurrence_filename,
            partition_keys=[partition_key],
            count_threshold=args.count_threshold,
        ),
    )

    co_occurrence.store_bundle(co_occurrence_filename, bundle)


def test_spaCy_co_occurrence_pipeline3():
    corpus_filename = './tests/test_data/legal_instrument_five_docs_test.zip'
    corpus_config: CorpusConfig = FakeSSI(
        source=corpus_filename,
        index_source='./tests/test_data/legal_instrument_five_docs_test.csv',
        # source='./tests/test_data/legal_instrument_corpus.zip',
        # index_source='./tests/test_data/legal_instrument_index.csv',
    )
    args = FakeComputeOptsSpacyCSV(
        corpus_filename=corpus_filename,
        corpus_tag="SATURNUS",
    )

    compute_co_occurrence(
        corpus_config=corpus_config,
        args=args,
        checkpoint_file='./tests/output/co_occurrence_checkpoint_pos.csv.zip',
    )
