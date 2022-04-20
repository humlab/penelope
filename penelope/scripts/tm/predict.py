import sys
from os.path import isfile, join, split

import click
from loguru import logger

import penelope.corpus as penelope
from penelope import pipeline
from penelope.scripts.utils import consolidate_cli_arguments, option2
from penelope.workflows.tm.predict import compute as workflow

# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('config-filename', required=True)
@click.argument('trained-model-folder', required=True)
# @click.argument('model-name', required=True)
@click.argument('target-folder', required=True)
@click.argument('target-name', required=False)
@option2('--options-filename')
@option2('--corpus-source', default=None)
@option2('--lemmatize/--no-lemmatize')
@option2('--pos-includes')
@option2('--pos-excludes')
@option2('--to-lower/--no-to-lower')
@option2('--min-word-length')
@option2('--max-word-length')
@option2('--keep-symbols/--no-keep-symbols')
@option2('--keep-numerals/--no-keep-numerals')
@option2('--remove-stopwords')
@option2('--only-alphabetic')
@option2('--only-any-alphanumeric')
@option2('--minimum-probability')
@option2('--n-tokens')
@option2('--force-checkpoint/--no-force-checkpoint')
@option2('--enable-checkpoint/--no-enable-checkpoint')
def click_main(
    options_filename: str = None,
    config_filename: str = None,
    corpus_source: str = None,
    trained_model_folder: str = None,
    target_folder: str = None,
    target_name: str = None,
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
    minimum_probability: float = 0.001,
    n_tokens: int = 200,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
):
    if not isfile(config_filename):
        click.echo(f"error: config file {config_filename} not found")
        sys.exit(1)

    if target_name is None:
        click.echo("error: TARGET_NAME not specified")
        sys.exit(1)

    arguments: dict = consolidate_cli_arguments(arguments=locals(), filename_key='options_filename')

    model_folder, model_name = split(trained_model_folder)

    arguments['model_folder'] = model_folder
    arguments['model_name'] = model_name

    if not isfile(join(model_folder, "model_options.json")):
        click.echo("error: no model in specified folder")
        sys.exit(1)

    main(**arguments)


def main(
    corpus_source: str = None,
    config_filename: str = None,
    model_folder: str = None,
    model_name: str = None,
    target_name: str = None,
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
    minimum_probability: float = 0.001,
    n_tokens: int = 200,
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

    extract_opts = penelope.ExtractTaggedTokensOpts(
        lemmatize=lemmatize,
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        **config.pipeline_payload.tagged_columns_names,
    )

    tag, folder = workflow(
        config=config,
        model_name=model_name,
        model_folder=model_folder,
        target_name=target_name,
        target_folder=target_folder,
        corpus_source=corpus_source,
        extract_opts=extract_opts,
        transform_opts=transform_opts,
        minimum_probability=minimum_probability,
        n_tokens=n_tokens,
        enable_checkpoint=enable_checkpoint,
        force_checkpoint=force_checkpoint,
    )

    logger.info(f"Done! Model {tag} stored in {folder}")


if __name__ == '__main__':
    click_main()  # pylint: disable=no-value-for-parameter
