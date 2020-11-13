from typing import Any

import click
from penelope.corpus.readers import AnnotationOpts
from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.utility import getLogger
from penelope.workflows import vectorize_corpus_workflow

logger = getLogger("penelope")
# pylint: disable=too-many-arguments, unused-argument


def split_filename(filename, sep='_'):
    parts = filename.replace('.', sep).split(sep)
    return parts


@click.command()
@click.argument('input_filename', type=click.STRING)  # , help='Model name.')
@click.argument('output_filename', type=click.STRING)  # , help='Model name.')
@click.argument('output-tag')
@click.option(
    '--corpus-type',
    default=None,
    type=click.Choice(['text', 'sparv4-csv']),
    help='Corpus type, only text and Sparv4 CSV currently supported',
)
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
@click.option('--doc-chunk-size', default=None, help='Split document in chunks of chunk-size words.', type=click.INT)
@click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
@click.option(
    '--only-alphabetic', default=False, is_flag=True, help='Keep only tokens having only alphabetic characters'
)
@click.option(
    '--only-any-alphanumeric', default=False, is_flag=True, help='Keep tokens with at least one alphanumeric char'
)
@click.option('--file-pattern', default='*.*', help='')
@click.option('-f', '--filename-field', default=None, help='Fields to extract from document name', multiple=True)
def main(
    input_filename: str = None,
    output_folder: str = None,
    output_tag: str = None,
    create_subfolder: bool = True,
    corpus_type: str = 'text',
    pos_includes: str = None,
    pos_excludes: str = '|MAD|MID|PAD|',
    lemmatize: bool = True,
    to_lowercase: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    doc_chunk_size: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    count_threshold: int = None,
    file_pattern: str = '*.*',
    filename_field: Any = None,
):

    tokens_transform_opts = TokensTransformOpts(
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
    )

    annotation_opts = None
    if corpus_type == 'sparv4-csv':
        file_pattern = '*.csv'
        annotation_opts = AnnotationOpts(
            pos_includes=pos_includes,
            pos_excludes=pos_excludes,
            lemmatize=lemmatize,
        )
    else:
        logger.info("PoS filter and lemmatize options not avaliable for raw text corpus")

    vectorize_corpus_workflow(
        corpus_type=corpus_type,
        input_filename=input_filename,
        output_folder=output_folder,
        output_tag=output_tag,
        create_subfolder=create_subfolder,
        filename_field=filename_field,
        filename_pattern=file_pattern,
        count_threshold=count_threshold,
        annotation_opts=annotation_opts,
        tokens_transform_opts=tokens_transform_opts,
    )


if __name__ == "__main__":
    main()
