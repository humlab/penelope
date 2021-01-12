import sys
from typing import Any, List, Sequence

import click
import penelope.notebook.co_occurrence.compute_corpus as workflow
import penelope.notebook.interface as interface
import penelope.pipeline as pipeline
from penelope.co_occurrence import ContextOpts, filename_to_folder_and_tag
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts

# pylint: disable=too-many-arguments


@click.command()
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
@click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
@click.option(
    '--only-alphabetic', default=False, is_flag=True, help='Keep only tokens having only alphabetic characters'
)
@click.option(
    '--only-any-alphanumeric', default=False, is_flag=True, help='Keep tokens with at least one alphanumeric char'
)
@click.option('-f', '--filename-field', default=None, help='Fields to extract from document name', multiple=True)
def main(
    input_filename: str,
    output_filename: str,
    count_threshold: int,
    concept: List[str],
    no_concept: bool,
    context_width: int,
    partition_key: Sequence[str],
    pos_includes: str,
    pos_excludes: str,
    lemmatize: bool,
    to_lowercase: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    filename_field: Any = None,
):

    try:
        corpus_folder, corpus_tag = filename_to_folder_and_tag(output_filename)

        args: interface.ComputeOpts = interface.ComputeOpts(
            corpus_type=pipeline.CorpusType.SparvCSV,
            corpus_filename=input_filename,
            target_folder=corpus_folder,
            corpus_tag=corpus_tag,
            tokens_transform_opts=TokensTransformOpts(
                to_lower=to_lowercase,
                to_upper=False,
                min_len=min_word_length,
                max_len=None,
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
            text_reader_opts=TextReaderOpts(
                filename_pattern='*.csv',
                filename_fields=filename_field,
                index_field=None,  # use filename
                as_binary=False,
            ),
            extract_tagged_tokens_opts=ExtractTaggedTokensOpts(
                pos_includes=pos_includes,
                pos_excludes=pos_excludes,
                lemmatize=lemmatize,
            ),
            count_threshold=count_threshold,
            create_subfolder=True,
            persist=True,
            context_opts=ContextOpts(
                context_width=context_width,
                concept=(concept or []),
                ignore_concept=no_concept,
            ),
            partition_keys=partition_key,
        )

        workflow.compute_co_occurrence(args)

    except Exception as ex:
        click.echo(ex)
        sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
