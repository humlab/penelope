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
@option2('--options-filename')
@option2('--corpus-source', default=None)
@option2('--target-folder', default=None)
@option2('--train-corpus-folder')
@option2('--lemmatize/--no-lemmatize')
@option2('--to-lower/--no-to-lower', default=False)
@option2('--pos-includes')
@option2('--pos-excludes')
@option2('--max-tokens')
@option2('--tf-threshold')
@option2('--n-topics')
@option2('--engine', default="gensim_lda-multicore")
@option2('--passes')
@option2('--alpha', default='asymmetric')
@option2('--random-seed')
@option2('--workers')
@option2('--max-iter')
@option2('--chunksize')
@option2('--update-every')
@option2('--store-corpus/--no-store-corpus')
@option2('--store-compressed/--no-store-compressed')
def click_main(
    options_filename: Optional[str] = None,
    target_name: Optional[str] = None,
    corpus_source: Optional[str] = None,
    config_filename: Optional[str] = None,
    train_corpus_folder: Optional[str] = None,
    target_folder: Optional[str] = None,
    to_lower: bool = True,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    max_tokens: int = None,
    tf_threshold: int = None,
    n_topics: int = 50,
    engine: str = "gensim_lda-multicore",
    passes: int = None,
    random_seed: int = None,
    alpha: str = 'asymmetric',
    workers: int = None,
    max_iter: int = None,
    chunksize: int = 2000,
    update_every: int = 1,
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
    to_lower: bool = True,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    max_tokens: int = None,
    tf_threshold: int = None,
    n_topics: int = 50,
    engine: str = "gensim_lda-multicore",
    passes: int = None,
    random_seed: int = None,
    alpha: str = 'asymmetric',
    workers: int = None,
    max_iter: int = None,
    chunksize: int = 2000,
    update_every: int = 1,
    store_corpus: bool = True,
    store_compressed: bool = True,
):
    to_lower = False  # for now...

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
    vectorize_opts: pc.VectorizeOpts = pc.VectorizeOpts(
        already_tokenized=True,
        lowercase=to_lower,
        max_tokens=max_tokens,
        min_tf=tf_threshold,
    )
    engine_args = remove_none(
        dict(
            n_topics=n_topics,
            passes=passes,
            random_seed=random_seed,
            alpha=alpha,
            workers=workers,
            max_iter=max_iter,
            work_folder=os.path.join(target_folder, target_name),
            chunksize=chunksize,
            update_every=update_every,
        )
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
        vectorize_opts=vectorize_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
    ).value()


if __name__ == '__main__':

    click_main()
