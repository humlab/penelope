import contextlib
import sys
from typing import Optional, Sequence

import click
from loguru import logger

import penelope.workflows.vectorize.dtm as workflow
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig
from penelope.pipeline.phrases import parse_phrases
from penelope.scripts.utils import consolidate_arguments, log_arguments, option2
from penelope.utility import pos_tags_to_str
from penelope.workflows import interface

# pylint: disable=too-many-arguments, unused-argument


@click.group()
def main():
    ...


@click.command()
@click.argument('config_filename', type=click.STRING)
@click.argument('input_filename', type=click.STRING)
@click.argument('output_folder', type=click.STRING)
@click.argument('output_tag', type=click.STRING)
@option2('--append-pos')
@option2('--deserialize-processes')
@option2('--filename-pattern')
@option2('--keep-numerals/--no-keep-numerals')
@option2('--keep-symbols/--no-keep-symbols')
@option2('--lemmatize/--no-lemmatize')
@option2('--max-tokens')
@option2('--max-word-length')
@option2('--min-word-length')
@option2('--options-filename')
@option2('--phrase-file')
@option2('--phrase')
@option2('--pos-excludes')
@option2('--pos-includes')
@option2('--pos-paddings')
@option2('--remove-stopwords')
@option2('--tf-threshold-mask')
@option2('--tf-threshold')
@option2('--only-alphabetic')
@option2('--only-any-alphanumeric')
@option2('--enable-checkpoint/--no-enable-checkpoint')
@option2('--force-checkpoint/--no-force-checkpoint')
@option2('--to-lower/--no-to-lower')
def to_dtm(
    options_filename: Optional[str] = None,
    config_filename: Optional[str] = None,
    input_filename: Optional[str] = None,
    output_folder: Optional[str] = None,
    output_tag: Optional[str] = None,
    filename_pattern: Optional[str] = None,
    phrase: Sequence[str] = None,
    phrase_file: Optional[str] = None,
    create_subfolder: bool = True,
    pos_includes: str = '',
    pos_paddings: str = '',
    pos_excludes: str = '',
    append_pos: bool = False,
    max_tokens: int = None,
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: Optional[str] = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    tf_threshold: int = 1,
    tf_threshold_mask: bool = False,
    deserialize_processes: int = 4,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
):
    arguments: dict = consolidate_arguments(arguments=locals(), filename_key='options_filename')

    process(**arguments)

    log_arguments(args=arguments, log_dir=output_folder)


@click.command()
@click.argument('options_filename', type=click.STRING)
def to_dtm_opts_file_only(options_filename: Optional[str] = None):
    arguments: dict = consolidate_arguments(arguments=locals(), filename_key='options_filename')

    process(**arguments)

    with contextlib.suppress(Exception):
        log_arguments(args=arguments, log_dir=arguments.get('output_folder'))


def process(
    config_filename: Optional[str] = None,
    input_filename: Optional[str] = None,
    output_folder: Optional[str] = None,
    output_tag: Optional[str] = None,
    filename_pattern: Optional[str] = None,
    phrase: Sequence[str] = None,
    phrase_file: Optional[str] = None,
    create_subfolder: bool = True,
    pos_includes: Optional[str] = None,
    pos_paddings: Optional[str] = None,
    pos_excludes: Optional[str] = None,
    append_pos: bool = False,
    max_tokens: int = None,
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: Optional[str] = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    tf_threshold: int = 1,
    tf_threshold_mask: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    deserialize_processes: int = 4,
):
    try:
        corpus_config: CorpusConfig = CorpusConfig.load(config_filename)
        phrases = parse_phrases(phrase_file, phrase)

        if pos_excludes is None:
            pos_excludes = pos_tags_to_str(corpus_config.pos_schema.Delimiter)

        if pos_paddings and pos_paddings.upper() in ["FULL", "ALL", "PASSTHROUGH"]:
            pos_paddings = pos_tags_to_str(corpus_config.pos_schema.all_types_except(pos_includes))
            logger.info(f"PoS paddings expanded to: {pos_paddings}")

        text_reader_opts: TextReaderOpts = corpus_config.text_reader_opts.copy()

        if filename_pattern is not None:
            text_reader_opts.filename_pattern = filename_pattern

        corpus_config.serialize_opts.deserialize_processes = max(1, deserialize_processes)

        tagged_columns: dict = corpus_config.pipeline_payload.tagged_columns_names
        args: interface.ComputeOpts = interface.ComputeOpts(
            corpus_type=corpus_config.corpus_type,
            corpus_source=input_filename,
            target_folder=output_folder,
            corpus_tag=output_tag,
            tf_threshold=tf_threshold,
            tf_threshold_mask=tf_threshold_mask,
            create_subfolder=create_subfolder,
            persist=True,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
            transform_opts=TokensTransformOpts(
                transforms={
                    'to-lower': to_lower,
                    'min-chars': min_word_length,
                    'max-chars': max_word_length,
                    'remove_stopwords': remove_stopwords,
                    'remove_numerals': not keep_numerals,
                    'remove_symbols': not keep_symbols,
                    'only-alphabetic': only_alphabetic,
                    'only-any-alphanumeric': only_any_alphanumeric,
                }
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
            vectorize_opts=VectorizeOpts(
                already_tokenized=True,
                min_tf=tf_threshold,
                max_tokens=max_tokens,
            ),
        )

        workflow.compute(opts=args, config=corpus_config)

        logger.info('Done!')

    except Exception as ex:  # pylint: disable=try-except-raise
        logger.exception(ex)
        click.echo(ex)
        sys.exit(1)


if __name__ == "__main__":
    # opts: dict = {
    #     'config_filename': './opts/inidun/dtm/curated_courier/v1.0.0/lemma_lowercase/corpus_config.yml',
    #     'input_filename': '/data/inidun/courier/corpus/v1.0.0/article_corpus.zip',
    #     'output_folder': '/data/inidun/courier/dtm/v1.0.0',
    #     'output_tag': 'curated_courier_v1.0.0_lowercase',
    #     # 'filename_pattern': '**/*.feather',
    #     'phrase': None,
    #     'phrase_file': None,
    #     'create_subfolder': True,
    #     'pos_includes': None,
    #     'pos_paddings': None,
    #     'pos_excludes': None,
    #     'append_pos': False,
    #     'max_tokens': 20000000,
    #     'to_lower': True,
    #     'lemmatize': False,
    #     'remove_stopwords': None,
    #     'min_word_length': 1,
    #     'max_word_length': None,
    #     'keep_symbols': True,
    #     'keep_numerals': True,
    #     'tf_threshold': 1,
    #     'tf_threshold_mask': True,
    #     'only_any_alphanumeric': False,
    #     'only_alphabetic': False,
    #     'enable_checkpoint': True,
    #     'force_checkpoint': False,
    #     'deserialize_processes': 1,
    # }
    # process(**opts)

    main.add_command(to_dtm, "args")
    main.add_command(to_dtm_opts_file_only, "opts-file")
    main()  # pylint: disable=no-value-for-parameter
