import os
import sys

import click
import penelope.corpus as penelope
from penelope import pipeline
from penelope.scripts.utils import option2, update_arguments_from_options_file

# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('config-filename', required=True)
@click.argument('model-folder', required=True)
# @click.argument('model-name', required=True)
@click.argument('target-folder', required=True)
@click.argument('target-name', required=False)
@option2('--options-filename', default=None)
@option2('--corpus-source', default=None)
@option2('--lemmatize/--no-lemmatize', default=True, is_flag=True)
@option2('--pos-includes', default='', type=click.STRING)
@option2('--pos-excludes', default='', type=click.STRING)
@option2('--to-lower/--no-to-lower', default=True, is_flag=True)
@option2('--min-word-length', default=1, type=click.IntRange(1, 99))
@option2('--max-word-length', default=None, type=click.IntRange(10, 99))
@option2('--keep-symbols/--no-keep-symbols', default=True, is_flag=True)
@option2('--keep-numerals/--no-keep-numerals', default=True, is_flag=True)
@option2('--remove-stopwords', default=None, type=click.Choice(['swedish', 'english']))
@option2('--only-alphabetic', default=False, is_flag=False)
@option2('--only-any-alphanumeric', default=False, is_flag=True)
@option2('--force-checkpoint/--no-force-checkpoint', default=False, is_flag=True)
@option2('--enable-checkpoint/--no-enable-checkpoint', default=True, is_flag=True)
def click_main(
    options_filename: str = None,
    config_filename: str = None,
    corpus_source: str = None,
    model_folder: str = None,
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
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
):
    arguments: dict = locals()

    if not os.path.isfile(os.path.join(model_folder, "model_options.json")):
        click.echo("error: no model in specified folder")
        sys.exit(1)

    model_folder, model_name = os.path.split(model_folder)

    arguments['model_folder'] = model_folder
    arguments['model_name'] = model_name
    arguments = update_arguments_from_options_file(arguments=arguments, filename_key='options_filename')

    if not os.path.isfile(config_filename):
        click.echo(f"error: file {config_filename} not found")
        sys.exit(1)

    if arguments.get('target_name') is None:
        click.echo("error: target_name not specified")
        sys.exit(1)

    _main(**arguments)


def _main(
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

    main(
        config=config,
        model_name=model_name,
        model_folder=model_folder,
        target_name=target_name,
        target_folder=target_folder,
        corpus_source=corpus_source,
        extract_opts=extract_opts,
        transform_opts=transform_opts,
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
    corpus_source: str = None,
    extract_opts: penelope.ExtractTaggedTokensOpts = None,
    transform_opts: penelope.TokensTransformOpts = None,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
):
    corpus_source: str = corpus_source or config.pipeline_payload.source

    if corpus_source is None:
        click.echo("usage: either corpus-folder or corpus filename must be specified")
        sys.exit(1)

    _: dict = (
        config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_source=corpus_source,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts,
            transform_opts=transform_opts,
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
