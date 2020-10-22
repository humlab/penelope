import click

import penelope.corpus.vectorizer as corpus_vectorizer


# pylint: disable=too-many-arguments
def split_filename(filename, sep='_'):
    parts = filename.replace('.', sep).split(sep)
    return parts


@click.command()
@click.argument('filename')
@click.argument('output-folder')
@click.option('--to-lower/--no-to-lower', '-l', default=True, help='Transform text to lower case.')
@click.option(
    '--remove-accents/--no-remove-accents', '-d', default=False, is_flag=True, help='Remove accents to lower case.'
)
@click.option('--min-length', default=2, help='Minimum length of words to keep', type=click.INT)
@click.option('--max-length', default=None, help='Maximum length of words to keep', type=click.INT)
@click.option('--doc-chunk-size', default=None, help='Split document in chunks of chunk-size words.', type=click.INT)
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='')
@click.option('--keep-symbols/--no-keep-symbols', default=False, is_flag=True, help='')
@click.option('--only-alphanumeric', default=False, is_flag=True, help='TBD.')
@click.option('--only-alphabetic', default=False, is_flag=True, help='')
@click.option('--file-pattern', default='*.txt', help='')
@click.option('--filename-field', '-f', help='RegExp fields to extract from document name', multiple=True)
def main(
    filename=None,
    output_folder=None,
    to_lower=True,
    remove_accents=False,
    min_length=2,
    max_length=None,
    doc_chunk_size=None,
    keep_numerals=False,
    keep_symbols=False,
    only_alphanumeric=True,
    only_alphabetic=True,
    file_pattern='*.txt',
    filename_field=None,
):

    kwargs = dict(
        to_lower=to_lower,
        remove_accents=remove_accents,
        min_len=min_length,
        max_len=max_length,
        doc_chunk_size=doc_chunk_size,
        keep_numerals=keep_numerals,
        keep_symbols=keep_symbols,
        only_any_alphanumeric=only_alphanumeric,
        only_alphabetic=only_alphabetic,
        pattern=file_pattern,
        filename_fields=filename_field,
    )

    corpus_vectorizer.generate_corpus(filename, output_folder=output_folder, **kwargs)


if __name__ == "__main__":
    main()
