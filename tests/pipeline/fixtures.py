import os

from penelope.co_occurrence import Bundle, ContextOpts
from penelope.corpus import TokensTransformOpts, VectorizeOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts
from penelope.notebook.interface import ComputeOpts
from penelope.pipeline import CorpusConfig, CorpusType
from penelope.pipeline.spacy.pipelines import spaCy_co_occurrence_pipeline
from penelope.utility import PropertyValueMaskingOpts


def FakeComputeOptsSpacyCSV(
    *,
    corpus_tag: str = 'MARS',
    corpus_filename: str = './tests/test_data/legal_instrument_five_docs_test.zip',
):  # pylint: disable=too-many-instance-attributes)

    return ComputeOpts(
        corpus_tag=corpus_tag,
        corpus_filename=corpus_filename,
        target_folder="./tests/output",
        corpus_type=CorpusType.SpacyCSV,
        # pos_scheme: PoS_Tag_Scheme = PoS_Tag_Schemes.Universal
        transform_opts=TokensTransformOpts(
            extra_stopwords=[],
            keep_numerals=True,
            keep_symbols=True,
            language='english',
            max_len=None,
            min_len=1,
            only_alphabetic=False,
            only_any_alphanumeric=False,
            remove_accents=False,
            remove_stopwords=True,
            stopwords=None,
            to_lower=True,
            to_upper=False,
        ),
        text_reader_opts=TextReaderOpts(
            filename_pattern='*.csv',
            filename_fields=['year:_:1'],
            index_field=None,  # use filename
            as_binary=False,
        ),
        extract_opts=ExtractTaggedTokensOpts(
            lemmatize=True,
            target_override=None,
            pos_includes='|NOUN|PROPN|VERB|',
            pos_paddings=None,
            pos_excludes='|PUNCT|EOL|SPACE|',
            passthrough_tokens=[],
            block_tokens=[],
            append_pos=False,
        ),
        filter_opts=PropertyValueMaskingOpts(
            is_alpha=False,
            is_punct=False,
            is_digit=None,
            is_stop=None,
            is_space=False,
        ),
        create_subfolder=False,
        persist=True,
        context_opts=ContextOpts(
            context_width=4,
            concept=set(),
            ignore_concept=False,
            partition_keys=['document_id'],
        ),
        count_threshold=1,
        vectorize_opts=VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            max_df=1.0,
            min_df=1,
            verbose=False,
        ),
    )


def FakeComputeOptsSparvCSV(
    *,
    corpus_tag: str = 'TELLUS',
    corpus_filename: str = './tests/test_data/tranströmer_corpus_export.sparv4.csv.zip',
) -> ComputeOpts:  # pylint: disable=too-many-instance-attributes)

    return ComputeOpts(
        corpus_tag=corpus_tag,
        corpus_filename=corpus_filename,
        target_folder="./tests/output",
        corpus_type=CorpusType.SparvCSV,
        transform_opts=TokensTransformOpts(
            to_lower=True,
            min_len=1,
            remove_stopwords=None,
            keep_symbols=True,
            keep_numerals=True,
            only_alphabetic=False,
            only_any_alphanumeric=False,
        ),
        text_reader_opts=TextReaderOpts(
            filename_pattern='*.csv',
            filename_fields=('year:_:1',),
            index_field=None,  # use filename
            as_binary=False,
        ),
        extract_opts=ExtractTaggedTokensOpts(
            pos_includes=None,
            pos_excludes='|MAD|MID|PAD|',
            pos_paddings=None,
            lemmatize=False,
        ),
        filter_opts=PropertyValueMaskingOpts(
            is_alpha=False,
            is_punct=False,
            is_digit=None,
            is_stop=None,
            is_space=False,
        ),
        create_subfolder=False,
        persist=True,
        context_opts=ContextOpts(
            concept=('jag',),
            context_width=2,
            partition_keys=['document_id'],
        ),
        count_threshold=1,
        vectorize_opts=VectorizeOpts(
            already_tokenized=True,
        ),
    )


def create_venus_bundle():
    """Note: Use the output from this test case to update the tests/test_data/VENUS test data"""

    config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI.yml')

    config.pipeline_payload.source = './tests/test_data/legal_instrument_five_docs_test.zip'
    config.pipeline_payload.document_index_source = './tests/test_data/legal_instrument_five_docs_test.csv'

    args = FakeComputeOptsSpacyCSV(
        corpus_tag="VENUS",
        corpus_filename=config.pipeline_payload.source,
    )
    args.target_folder = './tests/test_data/VENUS'
    args.context_opts = ContextOpts(context_width=4, ignore_concept=True, partition_keys=['document_id'])

    os.makedirs('./tests/test_data/VENUS', exist_ok=True)
    os.makedirs('./tests/output', exist_ok=True)

    checkpoint_filename: str = "./tests/output/co_occurrence_test_pos_csv.zip"

    bundle: Bundle = spaCy_co_occurrence_pipeline(
        corpus_config=config,
        corpus_filename=None,
        transform_opts=args.transform_opts,
        extract_opts=args.extract_opts,
        filter_opts=args.filter_opts,
        context_opts=args.context_opts,
        global_threshold_count=args.count_threshold,
        checkpoint_filename=checkpoint_filename,
    ).value()

    assert bundle.corpus is not None
    assert bundle.token2id is not None
    assert bundle.document_index is not None

    bundle.tag = args.corpus_tag
    bundle.folder = args.target_folder
    bundle.co_occurrences = bundle.corpus.to_co_occurrences(bundle.token2id)

    bundle.store()
