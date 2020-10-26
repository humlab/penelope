import json
import sys
from typing import Any, List, Tuple

import click

import penelope.co_occurrence as co_occurrence
from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus
from penelope.utility import replace_extension

# pylint: disable=too-many-arguments


# FIXME: #6 Add option for filtering out concept word
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
@click.option('-m', '--min-word-length', default=1, type=click.IntRange(1, 99), help='Min length of words to keep')
@click.option('-s', '--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('-n', '--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
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
    concept: List[str],
    no_concept: bool,
    count_threshold: int,
    context_width: int,
    partition_key: List[str],
    pos_includes: str,
    pos_excludes: str,
    lemmatize: bool,
    to_lowercase: bool,
    remove_stopwords: bool,
    min_word_length: int,
    keep_symbols: bool,
    keep_numerals: bool,
    only_alphabetic: bool,
    only_any_alphanumeric: bool,
    filename_field: Any,
):

    cli_concept_co_occurrence(
        input_filename=input_filename,
        output_filename=output_filename,
        concept=concept,
        no_concept=no_concept,
        count_threshold=count_threshold,
        context_width=context_width,
        partition_keys=partition_key,
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        lemmatize=lemmatize,
        to_lowercase=to_lowercase,
        remove_stopwords=remove_stopwords,
        min_word_length=min_word_length,
        keep_symbols=keep_symbols,
        keep_numerals=keep_numerals,
        only_alphabetic=only_alphabetic,
        only_any_alphanumeric=only_any_alphanumeric,
        filename_field=filename_field,
    )


def cli_concept_co_occurrence(
    input_filename: str,
    output_filename: str,
    *,
    concept: List[str] = None,
    no_concept: bool = False,
    count_threshold: int = None,
    context_width: int = None,
    partition_keys: Tuple[str, List[str]],
    pos_includes: str = None,
    pos_excludes: str = '|MAD|MID|PAD|',
    lemmatize: bool = True,
    to_lowercase: bool = True,
    remove_stopwords: bool = None,
    min_word_length: int = 1,
    keep_symbols: bool = True,
    keep_numerals: bool = True,
    only_alphabetic: bool = False,
    only_any_alphanumeric: bool = False,
    filename_field: Any = None,
):

    if len(concept or []) == 0:
        click.echo("please specify at least one concept (--concept e.g. --concept=information)")
        sys.exit(1)

    if len(filename_field or []) == 0:
        click.echo("please specify at least one filename field (--filename-field e.g. --filename-field='year:_:1')")
        sys.exit(1)

    if context_width is None:
        click.echo(
            "please specify at width of context as max distance from cencept (--context-width e.g. --context_width=2)"
        )
        sys.exit(1)

    if len(partition_keys or []) == 0:
        click.echo("please specify partition key(s) (--partition-key e.g --partition-key=year)")
        sys.exit(1)

    tokens_transform_opts = {
        'to_lower': to_lowercase,
        'to_upper': False,
        'min_len': min_word_length,
        'max_len': None,
        'remove_accents': False,
        'remove_stopwords': remove_stopwords is not None,
        'stopwords': None,
        'extra_stopwords': None,
        'language': remove_stopwords,
        'keep_numerals': keep_numerals,
        'keep_symbols': keep_symbols,
        'only_alphabetic': only_alphabetic,
        'only_any_alphanumeric': only_any_alphanumeric,
    }

    sparv_extract_opts = {'pos_includes': pos_includes, 'pos_excludes': pos_excludes, 'lemmatize': lemmatize}

    tokenizer_opts = {'filename_pattern': '*.csv', 'filename_fields': filename_field, 'as_binary': False}

    corpus = SparvTokenizedCsvCorpus(
        source=input_filename,
        **sparv_extract_opts,
        tokenizer_opts=tokenizer_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    co_occurrence.compute_and_store(
        corpus=corpus,
        concepts=concept,
        no_concept=no_concept,
        count_threshold=count_threshold,
        n_context_width=context_width,
        partition_keys=partition_keys,
        target_filename=output_filename,
    )
    with open(replace_extension(output_filename, 'json'), 'w') as json_file:
        store_options = {
            'input': input_filename,
            'output': output_filename,
            'concepts': list(concept),
            'no_concept': no_concept,
            'count_threshold': count_threshold,
            'n_context_width': context_width,
            'partition_keys': partition_keys,
            'tokenizer_opts': tokenizer_opts,
            'tokens_transform_opts': tokens_transform_opts,
            'sparv_extract_opts': sparv_extract_opts,
        }
        json.dump(store_options, json_file, indent=4)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
