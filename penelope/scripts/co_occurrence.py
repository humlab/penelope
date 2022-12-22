import os
from typing import List, Optional, Sequence

from loguru import logger

import penelope.workflows.co_occurrence as workflow
from penelope import utility as pu
from penelope.co_occurrence import ContextOpts, to_folder_and_tag
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig
from penelope.pipeline.phrases import parse_phrases
from penelope.scripts.utils import consolidate_cli_arguments, option2
from penelope.utility import pos_tags_to_str
from penelope.workflows import interface

try:
    import click
except ImportError:
    click = pu.DummyClass()


# pylint: disable=too-many-arguments, unused-argument


@click.command()
# @click.argument('corpus_config', type=click.STRING)
# @click.argument('input_filename', type=click.STRING)
# @click.argument('output_filename', type=click.STRING)
@option2('--options-filename')
@option2('--corpus-config')
@option2('--input-filename')
@option2('--output-filename')
@option2('--filename-pattern')
@option2('--concept')
@option2('--ignore-padding')
@option2('--windows-threshold')
@option2('--ignore-concept')
@option2('--context-width')
@option2('--compute-processes')
@option2('--compute-chunk-size')
@option2('--partition-key')
@option2('--lemmatize/--no-lemmatize')
@option2('--pos-includes')
@option2('--pos-paddings')
@option2('--pos-excludes')
@option2('--append-pos')
@option2('--phrase')
@option2('--phrase-file')
@option2('--to-lower/--no-to-lower')
@option2('--min-word-length')
@option2('--max-word-length')
@option2('--keep-symbols/--no-keep-symbols')
@option2('--keep-numerals/--no-keep-numerals')
@option2('--remove-stopwords')
@option2('--only-alphabetic')
@option2('--only-any-alphanumeric')
@option2('--tf-threshold')
@option2('--tf-threshold-mask')
@option2('--enable-checkpoint/--no-enable-checkpoint')
@option2('--force-checkpoint/--no-force-checkpoint')
@option2('--deserialize-processes')
def main(
    options_filename: Optional[str] = None,
    corpus_config: str = None,
    input_filename: str = None,
    output_filename: str = None,
    filename_pattern: str = None,
    concept: List[str] = None,
    ignore_concept: bool = False,
    windows_threshold: int = 1,
    ignore_padding: bool = False,
    context_width: int = None,
    compute_processes: int = None,
    compute_chunk_size: int = 10,
    partition_key: Sequence[str] = None,
    phrase: Sequence[str] = None,
    phrase_file: str = None,
    create_subfolder: bool = True,
    pos_includes: str = '',
    pos_paddings: str = '',
    pos_excludes: str = '',
    append_pos: bool = False,
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    tf_threshold: int = 1,
    tf_threshold_mask: bool = False,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    deserialize_processes: int = 4,
):
    try:
        arguments: dict = consolidate_cli_arguments(arguments=locals(), filename_key='options_filename')

        process_co_ocurrence(**arguments)
    except MissingOptionError as ex:
        print(ex)


class MissingOptionError(Exception):
    ...


def process_co_ocurrence(
    corpus_config: str = None,
    input_filename: str = None,
    output_filename: str = None,
    filename_pattern: str = None,
    concept: List[str] = None,
    ignore_concept: bool = False,
    ignore_padding: bool = False,
    context_width: int = None,
    windows_threshold: int = 1,
    compute_processes: int = None,
    compute_chunk_size: int = 10,
    partition_key: Sequence[str] = None,
    phrase: Sequence[str] = None,
    phrase_file: str = None,
    create_subfolder: bool = True,
    pos_includes: str = None,
    pos_paddings: str = None,
    pos_excludes: str = None,
    append_pos: bool = False,
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    tf_threshold: int = 1,
    tf_threshold_mask: bool = False,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    deserialize_processes: int = 4,
):
    try:

        if not output_filename:
            raise MissingOptionError("Output filename not specified")

        if not corpus_config:
            raise MissingOptionError("Corpus configuration YAML file not specified")

        output_folder, output_tag = to_folder_and_tag(output_filename)
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

        if not input_filename:
            input_filename = corpus_config.pipeline_payload.source

        corpus_config.folders(os.path.dirname(input_filename), 'replace')

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
                windows_threshold=windows_threshold,
            ),
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
        )

        workflow.compute(args=args, corpus_config=corpus_config)

        logger.info('Done!')

    except MissingOptionError:
        raise
    except Exception as ex:  # pylint: disable=try-except-raise, unused-variable
        logger.exception(ex)
        click.echo(ex)
        # sys.exit(1)
        raise


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

    # from click.testing import CliRunner

    # runner = CliRunner()
    # result = runner.invoke(
    #     main,
    #     ['--options-filename', 'opts/inidun/opts/20221213_co-occurrence.yml'],
    # )
    # print(result.output)
