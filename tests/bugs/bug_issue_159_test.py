from os.path import join as jj

import pytest
from loguru import logger

import penelope.workflows.co_occurrence as workflow
from penelope import corpus as pc
from penelope import pipeline as pp
from penelope.co_occurrence import Bundle, ContextOpts
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.phrases import parse_phrases
from penelope.pipeline.sparv.pipelines import to_tagged_frame_pipeline
from penelope.utility import pos_tags_to_str
from penelope.workflows.interface import ComputeOpts


@pytest.mark.skip(reason="Bug fixed")
def test_simple():
    config_filename = "tests/test_data/tranströmer.yml"
    input_filename: str = "tests/test_data/tranströmer_corpus_pos_csv.zip"
    corpus_config: pp.CorpusConfig = pp.CorpusConfig.load(config_filename)  # .folders(DATA_FOLDER)

    context_opts = ContextOpts(
        chunksize=10,
        concept=set(['träd']),
        context_width=2,
        ignore_concept=False,
        ignore_padding=True,
        partition_keys=['year'],
        processes=None,
    )
    transform_opts: pc.TokensTransformOpts = pc.TokensTransformOpts()
    extract_opts: pc.ExtractTaggedTokensOpts = pc.ExtractTaggedTokensOpts(
        pos_includes='',
        pos_excludes='MAD|MID|PAD',
        pos_paddings='PASSTHROUGH',
        lemmatize=True,
        append_pos=False,
        global_tf_threshold=2,
        global_tf_threshold_mask=True,
        **corpus_config.pipeline_payload.tagged_columns_names,
    )

    bundle: Bundle = (
        to_tagged_frame_pipeline(
            corpus_config=corpus_config,
            corpus_source=input_filename,
            enable_checkpoint=False,
            force_checkpoint=False,
        )
        .vocabulary(
            lemmatize=True,
            progress=True,
            tf_threshold=3,
            tf_keeps=context_opts.get_concepts(),
            close=True,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts,  # .clear_tf_threshold(),
            transform_opts=transform_opts,
        )
        .to_document_co_occurrence(
            context_opts=context_opts,
        )
        .to_corpus_co_occurrence(
            context_opts=context_opts,
        )
        .single()
        .content
    )

    assert bundle is not None


@pytest.mark.skip(reason="Bug fixed")
def test_tranströmer():  # pylint: disable=non-ascii-name
    config_filename = "tests/test_data/tranströmer.yml"
    input_filename: str = "tests/test_data/tranströmer_corpus_pos_csv.zip"
    corpus_config: pp.CorpusConfig = pp.CorpusConfig.load(config_filename)  # .folders(DATA_FOLDER)

    tf_threshold: int = 2
    context_opts = ContextOpts(
        chunksize=10,
        concept=set(['träd']),
        context_width=2,
        ignore_concept=False,
        ignore_padding=True,
        partition_keys=['year'],
        processes=None,
    )
    transform_opts: pc.TokensTransformOpts = pc.TokensTransformOpts()
    extract_opts: pc.ExtractTaggedTokensOpts = pc.ExtractTaggedTokensOpts(
        pos_includes='',
        pos_excludes='MAD|MID|PAD',
        pos_paddings='PASSTHROUGH',
        lemmatize=True,
        append_pos=False,
        global_tf_threshold=2,
        global_tf_threshold_mask=True,
        **corpus_config.pipeline_payload.tagged_columns_names,
    )

    bundle: Bundle = (
        to_tagged_frame_pipeline(
            corpus_config=corpus_config,
            corpus_source=input_filename,
            enable_checkpoint=False,
            force_checkpoint=False,
        )
        .vocabulary(
            lemmatize=True,
            progress=True,
            tf_threshold=tf_threshold,
            tf_keeps=context_opts.get_concepts(),
            close=True,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts,  # .clear_tf_threshold(),
            transform_opts=transform_opts,
        )
        .to_document_co_occurrence(
            context_opts=context_opts,
        )
        .to_corpus_co_occurrence(
            context_opts=context_opts,
        )
        .single()
        .content
    )

    assert bundle is not None


@pytest.mark.skip(reason="Bug fixed")
def test_bug():
    config_filename = "tests/test_data/tranströmer.yml"
    input_filename: str = "tests/test_data/tranströmer_corpus_pos_csv.zip"
    output_folder = 'tests/output'
    corpus_config = pp.CorpusConfig.load(config_filename)  # .folders(DATA_FOLDER)

    corpus_config.pipeline_payload.files(source=input_filename, document_index_source=None)
    corpus_config.checkpoint_opts.deserialize_processes = 1

    compute_opts = ComputeOpts(
        corpus_type=pp.CorpusType.SparvCSV,
        corpus_source=input_filename,
        target_folder=jj(output_folder, 'APA'),
        corpus_tag='APA',
        transform_opts=pc.TokensTransformOpts(transforms={'to-lower': True}),
        text_reader_opts=pc.TextReaderOpts(
            filename_pattern=None,
            filename_filter=None,
            filename_fields=[
                'year:prot\\_(\\d{4}).*',
                'year2:prot_\\d{4}(\\d{2})__*',
                'number:prot_\\d+[afk_]{0,4}__(\\d+).*',
            ],
            index_field=None,
            as_binary=False,
            sep='\t',
            quoting=3,
        ),
        extract_opts=pc.ExtractTaggedTokensOpts(
            pos_includes='VB',
            pos_excludes='',
            # pos_excludes='MAD|MID|PAD',
            # pos_paddings='AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO|VB',
            pos_paddings='PASSTHROUGH',
            lemmatize=True,
            append_pos=False,
            global_tf_threshold=2,
            global_tf_threshold_mask=True,
            **corpus_config.pipeline_payload.tagged_columns_names,
        ),
        vectorize_opts=pc.VectorizeOpts(
            already_tokenized=True,
            lowercase=True,
            stop_words=None,
            max_df=1.0,
            min_df=1,
            min_tf=1,
        ),
        tf_threshold=2,
        tf_threshold_mask=False,
        create_subfolder=True,
        persist=True,
        context_opts=ContextOpts(
            context_width=1,
            concept=set(['träd']),
            ignore_concept=False,
            partition_keys=['year'],
            processes=None,
            chunksize=10,
            ignore_padding=True,
        ),
        enable_checkpoint=False,
        force_checkpoint=False,
    )

    _ = workflow.compute(
        args=compute_opts,
        corpus_config=corpus_config,
        tagged_corpus_source=jj(output_folder, 'test.zip'),
    )


@pytest.mark.skip(reason="Bug fixed")
def test_process_co_ocurrence():
    concept: list[str] = ["sverige"]
    corpus_config: str = "tests/bugs/riksdagens-protokoll.yml"
    input_filename: str = "/data/westac/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip"

    append_pos: bool = False
    compute_chunk_size: int = 10
    compute_processes: int = None
    context_width: int = 1
    create_subfolder: bool = True
    deserialize_processes: int = 1
    enable_checkpoint: bool = True
    filename_pattern: str = None
    force_checkpoint: bool = False
    ignore_concept: bool = False
    ignore_padding: bool = True
    keep_numerals: bool = True
    keep_symbols: bool = True
    lemmatize: bool = True
    max_word_length: int = None
    min_word_length: int = 1
    only_alphabetic: bool = False
    only_any_alphanumeric: bool = False
    partition_key: str = "year"
    phrase: list[str] = []
    phrase_file: str = None
    pos_excludes: str = ''
    pos_includes: str = 'VB'
    pos_paddings: str = "PASSTHROUGH"
    remove_stopwords: bool = None
    tf_threshold: int = 2
    tf_threshold_mask: bool = True
    to_lower: bool = True

    output_folder, output_tag = "tests/output", "APA"
    corpus_config: CorpusConfig = CorpusConfig.load(corpus_config)
    phrases = parse_phrases(phrase_file, phrase)

    if pos_excludes is None:
        pos_excludes = pos_tags_to_str(corpus_config.pos_schema.Delimiter)

    if pos_paddings.upper() in ["FULL", "ALL", "PASSTHROUGH"]:
        pos_paddings = pos_tags_to_str(corpus_config.pos_schema.all_types_except(pos_includes))
        logger.info(f"PoS paddings expanded to: {pos_paddings}")

    text_reader_opts: pc.TextReaderOpts = corpus_config.text_reader_opts.copy()

    if filename_pattern is not None:
        text_reader_opts.filename_pattern = filename_pattern

    corpus_config.checkpoint_opts.deserialize_processes = max(1, deserialize_processes)

    tagged_columns: dict = corpus_config.pipeline_payload.tagged_columns_names
    args: ComputeOpts = ComputeOpts(
        corpus_type=corpus_config.corpus_type,
        corpus_source=input_filename,
        target_folder=output_folder,
        corpus_tag=output_tag,
        transform_opts=pc.TokensTransformOpts(
            transforms={
                'to-lower': to_lower,
                'min-chars': min_word_length,
                'max-chars': max_word_length,
                'remove-stopwords': remove_stopwords,
                'remove-numerals': not keep_numerals,
                'remove-symbols': not keep_symbols,
                'only-alphabetic': only_alphabetic,
                'only-any_alphanumeric': only_any_alphanumeric,
            }
        ),
        text_reader_opts=text_reader_opts,
        extract_opts=pc.ExtractTaggedTokensOpts(
            pos_includes=pos_includes,
            pos_paddings=pos_paddings,
            pos_excludes=pos_excludes,
            lemmatize=lemmatize,
            phrases=phrases,
            append_pos=append_pos,
            global_tf_threshold=tf_threshold,
            global_tf_threshold_mask=tf_threshold_mask,
            **tagged_columns,
        ),
        vectorize_opts=pc.VectorizeOpts(already_tokenized=True, max_tokens=None),
        tf_threshold=tf_threshold,
        tf_threshold_mask=tf_threshold_mask,
        create_subfolder=create_subfolder,
        persist=True,
        context_opts=ContextOpts(
            context_width=context_width,
            concept=set(concept or []),
            ignore_concept=ignore_concept,
            ignore_padding=ignore_padding,
            partition_keys=partition_key,
            processes=compute_processes,
            chunksize=compute_chunk_size,
        ),
        enable_checkpoint=enable_checkpoint,
        force_checkpoint=force_checkpoint,
    )

    workflow.compute(args=args, corpus_config=corpus_config)

    logger.info('Done!')


if __name__ == '__main__':
    test_simple()
