import json

import click
import penelope.corpus.sparv_corpus as sparv_corpus
from penelope.corpus.readers import ExtractTaggedTokensOpts
from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.utility import replace_extension, suffix_filename, timestamp_filename


# pylint: disable=too-many-arguments
def store_options_to_json_file(
    input_filename, output_filename, tokens_transform_opts: TokensTransformOpts, sparv_extract_opts
):

    store_options = {
        'input': input_filename,
        'output': output_filename,
        'tokens_transform_opts': tokens_transform_opts.props,
        'sparv_extract_opts': sparv_extract_opts,
    }

    store_options_filename = replace_extension(output_filename, 'json')
    with open(store_options_filename, 'w') as json_file:
        json.dump(store_options, json_file)


@click.command()
@click.argument('input_filename')  # , help='Model name.')
@click.option('--pos-includes', default='', help='List of POS tags to include e.g. "|NN|JJ|".')
@click.option('--pos-excludes', default='|MAD|MID|PAD|', help='List of POS tags to exclude e.g. "|MAD|MID|PAD|".')
@click.option('--chunk-size', 'chunk_size', default=None, help='Document chunk size, defult one.')
@click.option('--lemmatize/--no-lemmatize', default=True, is_flag=True, help='')
@click.option('--lower/--no-lower', default=True, is_flag=True, help='')
@click.option(
    '--remove-stopwords',
    default=None,
    type=click.Choice(['swedish', 'english']),
    help='Remove stopwords using given language',
)
@click.option('--min-word-length', default=None, type=click.IntRange(1, 99), help='Min length of words to keep')
@click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
@click.option('--version', 'version', default=4, help='Sparv version i.e. 3 or 4', type=click.IntRange(3, 4))
def prepare_train_corpus(
    input_filename,
    pos_includes,
    pos_excludes,
    chunk_size,
    lemmatize,
    lower,
    remove_stopwords,
    min_word_length,
    keep_symbols,
    keep_numerals,
    version,
):
    """Prepares the a training corpus from Sparv XML archive"""
    tokens_transform_opts: TokensTransformOpts = TokensTransformOpts(
        to_lower=lower,
        remove_stopwords=remove_stopwords is not None,
        language=remove_stopwords,
        min_len=min_word_length,
        max_len=None,
        keep_numerals=keep_numerals,
        keep_symbols=keep_symbols,
    )
    extract_tokens_opts = ExtractTaggedTokensOpts(
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        lemmatize=lemmatize,
    )
    output_filename = replace_extension(timestamp_filename(suffix_filename(input_filename, "text")), 'zip')

    reader_opts = {
        'chunk_size': chunk_size,
    }

    sparv_corpus.sparv_xml_extract_and_store(
        source=input_filename,
        target=output_filename,
        version=version,
        extract_tokens_opts=extract_tokens_opts,
        reader_opts=reader_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    store_options_to_json_file(
        input_filename,
        output_filename,
        tokens_transform_opts,
        dict(version=version, extract_tokens_opts=extract_tokens_opts, reader_opts=reader_opts),
    )


if __name__ == '__main__':
    prepare_train_corpus()  # pylint: disable=no-value-for-parameter
