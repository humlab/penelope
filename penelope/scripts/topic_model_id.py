import os
import sys
from typing import Optional

import click
import penelope.workflows.topic_model.tm_id as workflow
from penelope import corpus as pc
from penelope.scripts.utils import load_config, option2, remove_none, update_arguments_from_options_file

# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('config-filename', required=False)
@click.argument('target-name', required=False)
@option2('--options-filename', default=None)
@option2('--corpus-source', default=None)
@option2('--target-folder', default=None)
@option2('--train-corpus-folder', default=None, type=click.STRING)
@option2('--lemmatize/--no-lemmatize', default=True, is_flag=True)
@option2('--pos-includes', default='', type=click.STRING)
@option2('--pos-excludes', default='', type=click.STRING)
@option2('--n-topics', default=50, type=click.INT)
@option2('--engine', default="gensim_lda-multicore")
@option2('--passes', default=None, type=click.INT)
@option2('--alpha', default='asymmetric')
@option2('--random-seed', default=None, type=click.INT)
@option2('--workers', default=None, type=click.INT)
@option2('--max-iter', default=None, type=click.INT)
@option2('--store-corpus/--no-store-corpus', default=True, is_flag=True)
@option2('--store-compressed/--no-store-compressed', default=True, is_flag=True)
def click_main(
    options_filename: Optional[str] = None,
    target_name: Optional[str] = None,
    corpus_source: Optional[str] = None,
    config_filename: Optional[str] = None,
    train_corpus_folder: Optional[str] = None,
    target_folder: Optional[str] = None,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    n_topics: int = 50,
    engine: str = "gensim_lda-multicore",
    passes: int = None,
    random_seed: int = None,
    alpha: str = 'asymmetric',
    workers: int = None,
    max_iter: int = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
):
    """Create a topic model."""
    arguments: dict = update_arguments_from_options_file(arguments=locals(), filename_key='options_filename')
    main(**arguments)


def main(
    target_name: Optional[str] = None,
    corpus_source: Optional[str] = None,
    config_filename: Optional[str] = None,
    train_corpus_folder: Optional[str] = None,
    target_folder: Optional[str] = None,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    n_topics: int = 50,
    engine: str = "gensim_lda-multicore",
    passes: int = None,
    random_seed: int = None,
    alpha: str = 'asymmetric',
    workers: int = None,
    max_iter: int = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
):

    if not config_filename or not os.path.isfile(config_filename):
        click.echo("error: config file not specified/found")
        raise sys.exit(1)

    if target_name is None:
        click.echo("error: target_name not specified")
        raise sys.exit(1)

    config = load_config(config_filename, corpus_source)

    if corpus_source is None and config.pipeline_payload.source is None:
        click.echo("usage: corpus source must be specified")
        raise sys.exit(1)

    if not config.pipeline_key_exists("topic_modeling_pipeline"):
        click.echo("config error: `topic_modeling_pipeline` not specified")
        raise sys.exit(1)

    # transform_opts: pc.TokensTransformOpts = None

    extract_opts: pc.ExtractTaggedTokensOpts = pc.ExtractTaggedTokensOpts(
        lemmatize=lemmatize,
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        pos_column='pos_id',
        lemma_column='lemma_id',
        text_column='token_id',
    )

    engine_args = remove_none(
        {
            'n_topics': n_topics,
            'passes': passes,
            'random_seed': random_seed,
            'alpha': alpha,
            'workers': workers,
            'max_iter': max_iter,
            'work_folder': os.path.join(target_folder, target_name),
        }
    )
    # _: dict = config.get_pipeline(
    #     pipeline_key="topic_modeling_pipeline",

    _: dict = workflow.compute(
        corpus_config=config,
        corpus_source=corpus_source,
        target_name=target_name,
        target_folder=target_folder,
        train_corpus_folder=train_corpus_folder,
        extract_opts=extract_opts,
        # transform_opts=transform_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
    ).value()


if __name__ == '__main__':

    click_main()
