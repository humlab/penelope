import sys
from typing import List, Sequence

import click
import penelope.notebook.interface as interface
import penelope.workflows as workflows
from loguru import logger
from penelope.co_occurrence import ContextOpts, to_folder_and_tag
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig
from penelope.pipeline.convert import parse_phrases
from penelope.utility import PropertyValueMaskingOpts, pos_tags_to_str

# pylint: disable=too-many-arguments, unused-argument


@click.command()
@click.argument('corpus_config', type=click.STRING)
@click.argument('input_filename', type=click.STRING)  # , help='Model name.')
@click.argument('output_filename', type=click.STRING)  # , help='Model name.')
@click.option('-g', '--filename-pattern', default=None, help='Filename pattern', type=click.STRING)
@click.option('-c', '--concept', default=None, help='Concept', multiple=True, type=click.STRING)
@click.option('--no-concept', default=False, is_flag=True, help='Filter out concept word')
@click.option(
    '-w',
    '--context-width',
    default=None,
    help='Width of context on either side of concept. Window size = 2 * context_width + 1 ',
    type=click.INT,
)
@click.option('-p', '--partition-key', default=None, help='Partition key(s)', multiple=True, type=click.STRING)
@click.option(
    '-i', '--pos-includes', default=None, help='List of POS tags to include e.g. "|NN|JJ|".', type=click.STRING
)
@click.option(
    '-m', '--pos-paddings', default=None, help='List of POS tags to replace with a padding marker.', type=click.STRING
)
@click.option(
    '-x',
    '--pos-excludes',
    default=None,
    help='List of POS tags to exclude e.g. "|MAD|MID|PAD|".',
    type=click.STRING,
)
@click.option('-a', '--append-pos', default=False, is_flag=True, help='Append PoS to tokems')
@click.option('-m', '--phrase', default=None, help='Phrase', multiple=True, type=click.STRING)
@click.option('-n', '--phrase-file', default=None, help='Phrase filename', multiple=False, type=click.STRING)
@click.option('-b', '--lemmatize/--no-lemmatize', default=True, is_flag=True, help='Use word baseforms')
@click.option('-l', '--to-lowercase/--no-to-lowercase', default=True, is_flag=True, help='Lowercase words')
@click.option(
    '-r',
    '--remove-stopwords',
    default=None,
    type=click.Choice(['swedish', 'english']),
    help='Remove stopwords using given language',
)
@click.option(
    '--tf-threshold',
    default=1,
    type=click.IntRange(1, 99),
    help='Globoal TF threshold filter (words below filtered out)',
)
@click.option(
    '--tf-threshold-mask',
    default=False,
    is_flag=True,
    help='If true, then low TF words are kept, but masked as "__low_tf__"',
)
@click.option('--min-word-length', default=1, type=click.IntRange(1, 99), help='Min length of words to keep')
@click.option('--max-word-length', default=None, type=click.IntRange(10, 99), help='Max length of words to keep')
@click.option('--doc-chunk-size', default=None, help='Split document in chunks of chunk-size words.', type=click.INT)
@click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
@click.option(
    '--only-alphabetic', default=False, is_flag=True, help='Keep only tokens having only alphabetic characters'
)
@click.option(
    '--only-any-alphanumeric', default=False, is_flag=True, help='Keep tokens with at least one alphanumeric char'
)
@click.option('-f', '--force/--no-force', default=False, is_flag=True, help='Ignore checkpoints')
def main(
    corpus_config: str = None,
    input_filename: str = None,
    output_filename: str = None,
    filename_pattern: str = None,
    concept: List[str] = None,
    no_concept: bool = None,
    context_width: int = None,
    partition_key: Sequence[str] = None,
    phrase: Sequence[str] = None,
    phrase_file: str = None,
    create_subfolder: bool = True,
    pos_includes: str = None,
    pos_paddings: str = None,
    pos_excludes: str = None,
    append_pos: bool = False,
    to_lowercase: bool = True,
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
    force: bool = False,
):

    process_co_ocurrence(
        corpus_config=corpus_config,
        input_filename=input_filename,
        output_filename=output_filename,
        filename_pattern=filename_pattern,
        concept=concept,
        no_concept=no_concept,
        context_width=context_width,
        phrase=phrase,
        phrase_file=phrase_file,
        partition_key=partition_key,
        create_subfolder=create_subfolder,
        pos_includes=pos_includes,
        pos_paddings=pos_paddings,
        pos_excludes=pos_excludes,
        append_pos=append_pos,
        to_lowercase=to_lowercase,
        lemmatize=lemmatize,
        remove_stopwords=remove_stopwords,
        min_word_length=min_word_length,
        max_word_length=max_word_length,
        keep_symbols=keep_symbols,
        keep_numerals=keep_numerals,
        only_any_alphanumeric=only_any_alphanumeric,
        only_alphabetic=only_alphabetic,
        tf_threshold=tf_threshold,
        tf_threshold_mask=tf_threshold_mask,
        force=force,
    )


def process_co_ocurrence(
    corpus_config: str = None,
    input_filename: str = None,
    output_filename: str = None,
    filename_pattern: str = None,
    concept: List[str] = None,
    no_concept: bool = None,
    context_width: int = None,
    partition_key: Sequence[str] = None,
    phrase: Sequence[str] = None,
    phrase_file: str = None,
    create_subfolder: bool = True,
    pos_includes: str = None,
    pos_paddings: str = None,
    pos_excludes: str = None,
    append_pos: bool = False,
    to_lowercase: bool = True,
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
    force: bool = False,
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

        args: interface.ComputeOpts = interface.ComputeOpts(
            corpus_type=corpus_config.corpus_type,
            corpus_filename=input_filename,
            target_folder=output_folder,
            corpus_tag=output_tag,
            transform_opts=TokensTransformOpts(
                to_lower=to_lowercase,
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
                to_lowercase=to_lowercase,
                append_pos=append_pos,
                global_tf_threshold=tf_threshold,
                global_tf_threshold_mask=tf_threshold_mask,
            ),
            vectorize_opts=VectorizeOpts(already_tokenized=True),
            tf_threshold=tf_threshold,
            tf_threshold_mask=tf_threshold_mask,
            create_subfolder=create_subfolder,
            persist=True,
            context_opts=ContextOpts(
                context_width=context_width,
                concept=set(concept or []),
                ignore_concept=no_concept,
                partition_keys=partition_key,
            ),
            filter_opts=PropertyValueMaskingOpts(),
            force=force,
        )

        workflows.co_occurrence.compute(
            args=args,
            corpus_config=corpus_config,
        )

        logger.info('Done!')

    except Exception as ex:  # pylint: disable=try-except-raise
        logger.exception(ex)
        click.echo(ex)
        sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
