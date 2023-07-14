import os
import shutil
import uuid

from penelope.co_occurrence import Bundle, ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig, CorpusType
from penelope.pipeline.co_occurrence import to_co_occurrence_pipeline
from penelope.workflows.interface import ComputeOpts
from tests.co_occurrence.utils import create_simple_bundle_by_pipeline
from tests.fixtures import SIMPLE_CORPUS_ABCDEFG_7DOCS
from tests.notebook.co_occurrence.load_co_occurrences_gui_test import DATA_FOLDER
from tests.utils import OUTPUT_FOLDER

SPARV_TAGGED_COLUMNS: dict = dict(
    text_column='token',
    lemma_column='baseform',
    pos_column='pos',
)

SPACY_TAGGED_COLUMNS: dict = dict(
    text_column='text',
    lemma_column='lemma_',
    pos_column='pos_',
)


def ComputeOptsSpacyCSV(
    *,
    corpus_tag: str = 'MARS',
    corpus_source: str = './tests/test_data/SSI/legal_instrument_five_docs_test.zip',
) -> ComputeOpts:  # pylint: disable=too-many-instance-attributes)
    return ComputeOpts(
        corpus_tag=corpus_tag,
        corpus_source=corpus_source,
        target_folder="./tests/output",
        corpus_type=CorpusType.SpacyCSV,
        # pos_scheme: PoS_Tag_Scheme = PoS_Tag_Schemes.Universal
        transform_opts=TokensTransformOpts(dict(remove_stopwords='english', to_lower=True)),
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
            global_tf_threshold=1,
            global_tf_threshold_mask=False,
            **SPACY_TAGGED_COLUMNS,
            filter_opts=dict(
                is_alpha=False,
                is_punct=False,
                is_digit=None,
                is_stop=None,
                is_space=False,
            ),
        ),
        create_subfolder=False,
        persist=True,
        context_opts=ContextOpts(
            context_width=4,
            concept=set(),
            ignore_concept=False,
            partition_keys=['document_id'],
        ),
        tf_threshold=1,
        tf_threshold_mask=False,
        vectorize_opts=VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            max_df=1.0,
            min_df=1,
            min_tf=1,
            max_tokens=None,
        ),
    )


def ComputeOptsSparvCSV(
    *,
    corpus_tag: str = 'TELLUS',
    corpus_source: str = './tests/test_data/tranströmer/tranströmer_corpus_export.sparv4.csv.zip',
) -> ComputeOpts:  # pylint: disable=too-many-instance-attributes)
    return ComputeOpts(
        corpus_tag=corpus_tag,
        corpus_source=corpus_source,
        target_folder="./tests/output",
        corpus_type=CorpusType.SparvCSV,
        transform_opts=TokensTransformOpts(dict(to_lower=True)),
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
            **SPARV_TAGGED_COLUMNS,
            filter_opts=dict(
                is_alpha=False,
                is_punct=False,
                is_digit=None,
                is_stop=None,
                is_space=False,
            ),
        ),
        create_subfolder=False,
        persist=True,
        context_opts=ContextOpts(
            concept=('jag',),
            context_width=2,
            partition_keys=['document_id'],
        ),
        tf_threshold=1,
        tf_threshold_mask=False,
        vectorize_opts=VectorizeOpts(already_tokenized=True, min_tf=1, max_tokens=None),
    )


def create_bundle_by_spaCy_pipeline(config: CorpusConfig, context_opts: ContextOpts, tag: str):
    """Note: Use the output from this test case to update the tests/test_data/{tag} test data"""
    target_folder: str = f'./tests/test_data/{tag}'
    os.makedirs(target_folder, exist_ok=True)

    args = ComputeOptsSpacyCSV(
        corpus_tag=tag,
        corpus_source=config.pipeline_payload.source,
    )
    args.target_folder = target_folder
    args.context_opts = context_opts

    os.makedirs(target_folder, exist_ok=True)
    shutil.rmtree(target_folder, ignore_errors=True)

    tagged_corpus_source: str = f"./tests/output/{str(uuid.uuid1())}_test_pos_csv.zip"

    bundle: Bundle = to_co_occurrence_pipeline(
        corpus_config=config,
        corpus_source=None,
        transform_opts=args.transform_opts,
        extract_opts=args.extract_opts,
        context_opts=args.context_opts,
        tf_threshold=args.tf_threshold,
        tagged_corpus_source=tagged_corpus_source,
    ).value()

    assert bundle.corpus is not None
    assert bundle.token2id is not None
    assert bundle.document_index is not None

    bundle.tag = args.corpus_tag
    bundle.folder = args.target_folder
    bundle.co_occurrences = bundle.corpus.to_co_occurrences(bundle.token2id)

    return bundle


def create_test_data_bundles():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI/SSI.yml')

    config.pipeline_payload.source = './tests/test_data/SSI/legal_instrument_five_docs_test.zip'
    config.pipeline_payload.document_index_source = './tests/test_data/SSI/legal_instrument_five_docs_test.csv'

    tag: str = 'VENUS-CONCEPT'
    create_bundle_by_spaCy_pipeline(
        config=config,
        context_opts=ContextOpts(
            context_width=4, concept={"cultural"}, ignore_concept=True, partition_keys=['document_id']
        ),
        tag=tag,
    ).store()

    tag: str = 'VENUS'
    create_bundle_by_spaCy_pipeline(
        config=config,
        context_opts=ContextOpts(context_width=4, concept={}, ignore_concept=True, partition_keys=['document_id']),
        tag=tag,
    ).store()

    tag: str = 'ABCDEFG_7DOCS'
    create_bundle_by_spaCy_pipeline(
        config=config,
        context_opts=ContextOpts(context_width=4, concept={}, ignore_concept=True, partition_keys=['document_id']),
        tag=tag,
    ).store()

    tag: str = 'ABCDEFG_7DOCS'
    create_simple_bundle_by_pipeline(
        data=SIMPLE_CORPUS_ABCDEFG_7DOCS,
        context_opts=ContextOpts(concept={}, ignore_concept=False, context_width=2),
        tag=tag,
        folder=DATA_FOLDER,
    ).store()

    tag: str = 'ABCDEFG_7DOCS_CONCEPT'
    create_simple_bundle_by_pipeline(
        data=SIMPLE_CORPUS_ABCDEFG_7DOCS,
        context_opts=ContextOpts(concept={'g'}, ignore_concept=False, context_width=2),
        tag=tag,
        folder=DATA_FOLDER,
    ).store()
