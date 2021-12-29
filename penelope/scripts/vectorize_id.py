import sys
from typing import Optional, Sequence

import click
import penelope.workflows.vectorize.dtm_id as workflow
from loguru import logger
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig
from penelope.pipeline.phrases import parse_phrases
from penelope.scripts.utils import option2, update_arguments_from_options_file
from penelope.utility import pos_tags_to_str

# pylint: disable=too-many-arguments, unused-argument, useless-super-delegation


@click.command()
@click.argument('config_filename', type=click.STRING, required=False)
@click.argument('corpus_source', type=click.STRING, required=False)
@click.argument('output_folder', type=click.STRING, required=False)
@click.argument('output_tag', type=click.STRING, required=False)
@option2('--options-filename', type=click.STRING, default=None)
@option2('--filename-pattern', default=None, type=click.STRING)
@option2('--pos-includes', default='', type=click.STRING)
@option2('--pos-paddings', default='', type=click.STRING)
@option2('--pos-excludes', default='', type=click.STRING)
@option2('--append-pos', default=False, is_flag=True)
@option2('--phrase', default=None, multiple=True, type=click.STRING)
@option2('--phrase-file', default=None, multiple=False, type=click.STRING)
@option2('--lemmatize/--no-lemmatize', default=True, is_flag=True)
@option2('--to-lower/--no-to-lower', default=True, is_flag=True)
@option2('--remove-stopwords', default=None, type=click.Choice(['swedish', 'english']))
@option2('--tf-threshold', default=1, type=click.IntRange(1, 99))
@option2('--tf-threshold-mask', default=False, is_flag=True)
@option2('--min-word-length', default=1, type=click.IntRange(1, 99))
@option2('--max-word-length', default=None, type=click.IntRange(10, 99))
@option2('--keep-symbols/--no-keep-symbols', default=True, is_flag=True)
@option2('--keep-numerals/--no-keep-numerals', default=True, is_flag=True)
@option2('--deserialize-processes', default=4, type=click.IntRange(1, 99))
def main(
    options_filename: Optional[str] = None,
    config_filename: Optional[str] = None,
    corpus_source: Optional[str] = None,
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
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: Optional[str] = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    tf_threshold: int = 1,
    tf_threshold_mask: bool = False,
    deserialize_processes: int = 4,
):
    arguments: dict = update_arguments_from_options_file(arguments=locals(), filename_key='options_filename')
    process(**arguments)


def process(
    config_filename: Optional[str] = None,
    corpus_source: Optional[str] = None,
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
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: Optional[str] = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    tf_threshold: int = 1,
    tf_threshold_mask: bool = False,
    deserialize_processes: int = 4,
):

    try:
        corpus_config: CorpusConfig = CorpusConfig.load(config_filename).folders(corpus_source, method='replace')
        phrases: dict = parse_phrases(phrase_file, phrase)

        if pos_excludes is None:
            pos_excludes = pos_tags_to_str(corpus_config.pos_schema.Delimiter)

        if pos_paddings and pos_paddings.upper() in ["FULL", "ALL", "PASSTHROUGH"]:
            pos_paddings = pos_tags_to_str(corpus_config.pos_schema.all_types_except(pos_includes))
            logger.info(f"PoS paddings expanded to: {pos_paddings}")

        text_reader_opts: TextReaderOpts = corpus_config.text_reader_opts.copy()

        if filename_pattern is not None:
            text_reader_opts.filename_pattern = filename_pattern

        corpus_config.checkpoint_opts.deserialize_processes = max(1, deserialize_processes)

        tagged_columns: dict = corpus_config.pipeline_payload.tagged_columns_names
        args: workflow.ComputeOpts = workflow.ComputeOpts(
            corpus_type=corpus_config.corpus_type,
            corpus_source=corpus_source,
            target_folder=output_folder,
            corpus_tag=output_tag,
            tf_threshold=tf_threshold,
            tf_threshold_mask=tf_threshold_mask,
            create_subfolder=create_subfolder,
            persist=True,
            filename_pattern=filename_pattern,
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
            ),
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
        )

        workflow.compute(args=args, corpus_config=corpus_config)

        logger.info('Done!')

    except Exception as ex:  # pylint: disable=try-except-raise
        logger.exception(ex)
        click.echo(ex)
        sys.exit(1)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
