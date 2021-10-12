import os
import sys

import click
import penelope.corpus as penelope
import yaml
from penelope import pipeline
from penelope.utility.pandas_utils import PropertyValueMaskingOpts

# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('config-filename', required=True)
@click.argument('model-folder', required=True)
# @click.argument('model-name', required=True)
@click.argument('target-folder', required=True)
@click.argument('target-name', required=False)
@click.option('--options-filename', default=None, help='Use values in YAML file as command line options.')
@click.option('--corpus-folder', default=None, help='Corpus folder (if vectorized corpus exists on disk).')
@click.option('--corpus-filename', default=None, help='Corpus filename (overrides config)')
@click.option('-b', '--lemmatize/--no-lemmatize', default=True, is_flag=True, help='Use word baseforms')
@click.option('-i', '--pos-includes', default='', help='POS tags to include e.g. "|NN|JJ|".', type=click.STRING)
@click.option('-x', '--pos-excludes', default='', help='POS tags to exclude e.g. "|MAD|MID|PAD|".', type=click.STRING)
@click.option('-l', '--to-lower/--no-to-lower', default=True, is_flag=True, help='Lowercase words')
@click.option('--min-word-length', default=1, type=click.IntRange(1, 99), help='Min length of words to keep')
@click.option('--max-word-length', default=None, type=click.IntRange(10, 99), help='Max length of words to keep')
@click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
@click.option('--remove-stopwords', default=None, type=click.Choice(['swedish', 'english']), help='Remove stopwords')
@click.option('--only-alphabetic', default=False, is_flag=False, help='Remove tokens with non-alphabetic character(s)')
@click.option('--only-any-alphanumeric', default=False, is_flag=True, help='Remove tokes with no alphanumeric char')
@click.option('--force-checkpoint/--no-force-checkpoint', default=False, is_flag=True, help='')
@click.option('--enable-checkpoint/--no-enable-checkpoint', default=True, is_flag=True, help='')
def click_main(
    config_filename: str = None,
    model_folder: str = None,
    # model_name: str = None,
    target_folder: str = None,
    target_name: str = None,
    options_filename: str = None,
    corpus_filename: str = None,
    corpus_folder: str = None,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    to_lower: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
):
    arguments: dict = locals()
    del arguments['options_filename']

    if not os.path.isfile(os.path.join(model_folder, "model_options.json")):
        click.echo("error: no model in specified folder")
        sys.exit(1)

    model_folder, model_name = os.path.split(model_folder)

    arguments['model_folder'] = model_folder
    arguments['model_name'] = model_name

    if options_filename is not None:
        with open(options_filename, "r") as fp:
            options: dict = yaml.load(fp, Loader=yaml.FullLoader)
        arguments.update(options)

    if not os.path.isfile(config_filename):
        click.echo(f"error: file {config_filename} not found")
        sys.exit(1)

    if arguments.get('target_name') is None:
        click.echo("error: target_name not specified")
        sys.exit(1)

    _main(**arguments)


def _main(
    model_folder: str = None,
    model_name: str = None,
    config_filename: str = None,
    target_name: str = None,
    corpus_filename: str = None,
    corpus_folder: str = None,
    target_folder: str = None,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    to_lower: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
):
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(path=config_filename)

    transform_opts: penelope.TokensTransformOpts = penelope.TokensTransformOpts(
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
        only_alphabetic=only_alphabetic,
        only_any_alphanumeric=only_any_alphanumeric,
    )

    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts()
    extract_opts = penelope.ExtractTaggedTokensOpts(
        lemmatize=lemmatize,
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        **config.pipeline_payload.tagged_columns_names,
    )

    main(
        config=config,
        model_name=model_name,
        model_folder=model_folder,
        target_name=target_name,
        target_folder=target_folder,
        corpus_filename=corpus_filename,
        corpus_folder=corpus_folder,
        extract_opts=extract_opts,
        transform_opts=transform_opts,
        filter_opts=filter_opts,
        enable_checkpoint=enable_checkpoint,
        force_checkpoint=force_checkpoint,
    )


def main(
    *,
    config: pipeline.CorpusConfig,
    model_folder: str = None,
    model_name: str = None,
    target_folder: str = None,
    target_name: str,
    corpus_filename: str = None,
    corpus_folder: str = None,
    extract_opts: penelope.ExtractTaggedTokensOpts = None,
    transform_opts: penelope.TokensTransformOpts = None,
    filter_opts: PropertyValueMaskingOpts = None,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
):
    corpus_filename: str = corpus_filename or config.pipeline_payload.source

    if corpus_filename is None and corpus_folder is None:
        click.echo("usage: either corpus-folder or corpus filename must be specified")
        sys.exit(1)

    if corpus_folder is None:
        corpus_folder, _ = os.path.split(os.path.abspath(corpus_filename))

    _: dict = (
        config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_filename=corpus_filename,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts,
            transform_opts=transform_opts,
            filter_opts=filter_opts,
        )
        .predict_topics(
            model_folder=model_folder,
            model_name=model_name,
            target_folder=target_folder,
            target_name=target_name,
        )
    ).value()


if __name__ == '__main__':
    click_main()  # pylint: disable=no-value-for-parameter
