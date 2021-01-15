import sys
from typing import List, Sequence

import click
import penelope.notebook.interface as interface
import penelope.workflows as workflows
from penelope.co_occurrence import ContextOpts, filename_to_folder_and_tag
from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig
from penelope.utility import getLogger

logger = getLogger("penelope")
# pylint: disable=too-many-arguments


@click.command()
@click.argument('corpus_config', type=click.STRING)  # , help='Model name.')
@click.argument('input_filename', type=click.STRING)  # , help='Model name.')
@click.argument('output_filename', type=click.STRING)  # , help='Model name.')
@click.option('-c', '--concept', default=None, help='Concept', multiple=True, type=click.STRING)
@click.option('--no-concept', default=False, is_flag=True, help='Filter out concept word')
@click.option('--count-threshold', default=None, help='Filter out co_occurrences below threshold', type=click.INT)
@click.option(
    '-w',
    '--context-width',
    default=None,
    help='Width of context on either side of concept. Window size = 2 * context_width + 1 ',
    type=click.INT,
)
@click.option('-p', '--partition-key', default=None, help='Pertition key(s)', multiple=True, type=click.STRING)
@click.option(
    '-i', '--pos-includes', default=None, help='List of POS tags to include e.g. "|NN|JJ|".', type=click.STRING
)
@click.option(
    '-x',
    '--pos-excludes',
    default='|MAD|MID|PAD|',
    help='List of POS tags to exclude e.g. "|MAD|MID|PAD|".',
    type=click.STRING,
)
@click.option('-b', '--lemmatize/--no-lemmatize', default=True, is_flag=True, help='Use word baseforms')
@click.option('-l', '--to-lowercase/--no-to-lowercase', default=True, is_flag=True, help='Lowercase words')
@click.option(
    '-r',
    '--remove-stopwords',
    default=None,
    type=click.Choice(['swedish', 'english']),
    help='Remove stopwords using given language',
)
@click.option('--min-word-length', default=1, type=click.IntRange(1, 99), help='Min length of words to keep')
@click.option('--max-word-length', default=None, type=click.IntRange(10, 99), help='Max length of words to keep')
@click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
@click.option(
    '--only-alphabetic', default=False, is_flag=True, help='Keep only tokens having only alphabetic characters'
)
@click.option(
    '--only-any-alphanumeric', default=False, is_flag=True, help='Keep tokens with at least one alphanumeric char'
)
def main(
    corpus_config: str = None,
    input_filename: str = None,
    output_filename: str = None,
    concept: List[str] = None,
    no_concept: bool = None,
    context_width: int = None,
    partition_key: Sequence[str] = None,
    create_subfolder: bool = True,
    pos_includes: str = None,
    pos_excludes: str = '|MAD|MID|PAD|',
    to_lowercase: bool = True,
    lemmatize: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    count_threshold: int = None,
):

    try:
        output_folder, output_tag = filename_to_folder_and_tag(output_filename)
        corpus_config: CorpusConfig = CorpusConfig.load(corpus_config)

        args: interface.ComputeOpts = interface.ComputeOpts(
            corpus_type=corpus_config.corpus_type,
            corpus_filename=input_filename,
            target_folder=output_folder,
            corpus_tag=output_tag,
            tokens_transform_opts=TokensTransformOpts(
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
            text_reader_opts=corpus_config.text_reader_opts,
            extract_tagged_tokens_opts=ExtractTaggedTokensOpts(
                pos_includes=pos_includes,
                pos_excludes=pos_excludes,
                lemmatize=lemmatize,
            ),
            vectorize_opts=VectorizeOpts(already_tokenized=True),
            count_threshold=count_threshold,
            create_subfolder=create_subfolder,
            persist=True,
            context_opts=ContextOpts(
                context_width=context_width,
                concept=(concept or []),
                ignore_concept=no_concept,
            ),
            partition_keys=partition_key,
        )

        workflows.co_occurrence.compute(
            args=args,
            corpus_config=corpus_config,
        )

        logger.info('Done!')

    except Exception as ex:
        click.echo(ex)
        sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
