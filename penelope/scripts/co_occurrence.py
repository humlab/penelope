from typing import List, Optional, Sequence

import click
import penelope.workflows.co_occurrence as workflow
from loguru import logger
from penelope.co_occurrence import ContextOpts, to_folder_and_tag
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig
from penelope.pipeline.phrases import parse_phrases
from penelope.scripts.utils import option2, update_arguments_from_options_file
from penelope.utility import pos_tags_to_str
from penelope.workflows import interface

# pylint: disable=too-many-arguments, unused-argument


@click.command()
@click.argument('corpus_config', type=click.STRING)
@click.argument('input_filename', type=click.STRING)
@click.argument('output_filename', type=click.STRING)
@option2('--options-filename', default=None)
@option2('--filename-pattern', default=None, type=click.STRING)
@option2('--concept', default=None, multiple=True, type=click.STRING)
@option2('--ignore-padding', default=False, is_flag=True)
@option2('--ignore-concept', default=False, is_flag=True)
@option2('--context-width', default=None, type=click.INT)
@option2('--compute-processes', default=None, type=click.INT)
@option2('--compute-chunksize', default=10, type=click.INT)
@option2('--partition-key', default=None, multiple=True, type=click.STRING)
@option2('--lemmatize/--no-lemmatize', default=True, is_flag=True)
@option2('--pos-includes', default='', type=click.STRING)
@option2('--pos-paddings', default='', type=click.STRING)
@option2('--pos-excludes', default='', type=click.STRING)
@option2('--append-pos', default=False, is_flag=True)
@option2('--phrase', default=None, multiple=True, type=click.STRING)
@option2('--phrase-file', default=None, multiple=False, type=click.STRING)
@option2('--to-lower/--no-to-lower', default=True, is_flag=True)
@option2('--min-word-length', default=1, type=click.IntRange(1, 99))
@option2('--max-word-length', default=None, type=click.IntRange(10, 99))
@option2('--keep-symbols/--no-keep-symbols', default=True, is_flag=True)
@option2('--keep-numerals/--no-keep-numerals', default=True, is_flag=True)
@option2('--remove-stopwords', default=None, type=click.Choice(['swedish', 'english']))
@option2('--only-alphabetic', default=False, is_flag=False)
@option2('--only-any-alphanumeric', default=False, is_flag=True)
@option2('--tf-threshold', default=1, type=click.IntRange(1, 99))
@option2('--tf-threshold-mask', default=False, is_flag=True)
@option2('--doc-chunk-size', default=None, type=click.INT)
@option2('--enable-checkpoint/--no-enable-checkpoint', default=True, is_flag=True)
@option2('--force-checkpoint/--no-force-checkpoint', default=False, is_flag=True)
@option2('--deserialize-processes', default=4, type=click.IntRange(1, 99))
def main(
    options_filename: Optional[str] = None,
    corpus_config: str = None,
    input_filename: str = None,
    output_filename: str = None,
    filename_pattern: str = None,
    concept: List[str] = None,
    ignore_concept: bool = False,
    ignore_padding: bool = False,
    context_width: int = None,
    compute_processes: int = None,
    compute_chunksize: int = 10,
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
    doc_chunk_size: int = None,
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
    arguments: dict = update_arguments_from_options_file(arguments=locals(), filename_key='options_filename')

    process_co_ocurrence(**arguments)


def process_co_ocurrence(
    corpus_config: str = None,
    input_filename: str = None,
    output_filename: str = None,
    filename_pattern: str = None,
    concept: List[str] = None,
    ignore_concept: bool = False,
    ignore_padding: bool = False,
    context_width: int = None,
    compute_processes: int = None,
    compute_chunksize: int = 10,
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
    doc_chunk_size: int = None,
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
            vectorize_opts=VectorizeOpts(already_tokenized=True),
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
                chunksize=compute_chunksize,
            ),
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
        )

        workflow.compute(args=args, corpus_config=corpus_config)

        logger.info('Done!')

    except Exception as ex:  # pylint: disable=try-except-raise, unused-variable
        logger.exception(ex)
        click.echo(ex)
        # sys.exit(1)
        raise


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
