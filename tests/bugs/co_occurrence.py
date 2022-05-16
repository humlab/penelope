from loguru import logger

import penelope.workflows.co_occurrence as workflow
from penelope.co_occurrence import ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig
from penelope.pipeline.phrases import parse_phrases
from penelope.utility import pos_tags_to_str
from penelope.workflows import interface

# pylint: disable=too-many-arguments, unused-argument


def process_co_ocurrence():
    concept: list[str] = ["sverige"]
    # corpus_config: str = "tests/bugs/riksdagens-protokoll.yml"
    # input_filename: str = "/data/westac/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip"

    corpus_config: str = "tests/test_data/tranströmer.yml"
    input_filename: str = "tests/test_data/tranströmer_corpus_pos_csv.zip"

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

    text_reader_opts: TextReaderOpts = corpus_config.text_reader_opts.copy()

    if filename_pattern is not None:
        text_reader_opts.filename_pattern = filename_pattern

    corpus_config.checkpoint_opts.deserialize_processes = max(1, deserialize_processes)

    tagged_columns: dict = corpus_config.pipeline_payload.tagged_columns_names
    args: interface.ComputeOpts = interface.ComputeOpts(
        corpus_type=corpus_config.corpus_type,
        corpus_source=input_filename,
        target_folder=output_folder,
        corpus_tag=output_tag,
        transform_opts=TokensTransformOpts(
            to_lower=to_lower,
            to_upper=False,
            min_len=min_word_length,
            max_len=max_word_length,
            remove_accents=False,
            remove_stopwords=(remove_stopwords is not None),
            stopwords=None,
            extra_stopwords=None,
            language=remove_stopwords,
            keep_numerals=keep_numerals,
            keep_symbols=keep_symbols,
            only_alphabetic=only_alphabetic,
            only_any_alphanumeric=only_any_alphanumeric,
        ),
        text_reader_opts=text_reader_opts,
        extract_opts=ExtractTaggedTokensOpts(
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
        vectorize_opts=VectorizeOpts(already_tokenized=True, max_tokens=None),
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
    process_co_ocurrence()
