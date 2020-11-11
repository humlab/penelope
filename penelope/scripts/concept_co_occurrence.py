import sys
from typing import Any, List

import click
from penelope.co_occurrence.concept_co_occurrence import ConceptContextOpts
from penelope.corpus.readers import AnnotationOpts
from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.workflows import WorkflowException, concept_co_occurrence_workflow

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
@click.option(
    '-v',
    '--store-vectorized',
    default=True,
    is_flag=True,
    help='Stores co-occurrence token-pairs as a vectorized corpus normalized by yearly total token count',
    multiple=True,
)
def main(
    input_filename: str,
    output_filename: str,
    concept: List[str],
    no_concept: bool,
    count_threshold: int,
    context_width: int,
    partition_key: List[str],
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
    store_vectorized: bool = True,
):

    tokens_transform_opts = TokensTransformOpts(
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
    )
    annotation_opts = AnnotationOpts(
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        lemmatize=lemmatize,
    )
    concept_opts = ConceptContextOpts(
        concept=concept,
        ignore_concept=no_concept,
        context_width=context_width,
    )
    try:

        concept_co_occurrence_workflow(
            input_filename=input_filename,
            output_filename=output_filename,
            count_threshold=count_threshold,
            filename_field=filename_field,
            store_vectorized=store_vectorized,
            partition_keys=partition_key,
            concept_opts=concept_opts,
            annotation_opts=annotation_opts,
            tokens_transform_opts=tokens_transform_opts,
        )

    except WorkflowException as ex:
        click.echo(ex)
        sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
