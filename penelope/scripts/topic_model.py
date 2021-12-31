import os
import sys
from typing import Optional

import click
import penelope.corpus as pc
from penelope import pipeline
from penelope.scripts.utils import load_config, option2, remove_none, update_arguments_from_options_file

# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('config-filename', required=True)
@click.argument('target-name', required=False)
@option2('--options-filename', default=None)
@option2('--corpus-source', default=None)
@option2('--target-folder', default=None)
@option2('--train-corpus-folder')
@option2('--fix-hyphenation/--no-fix-hyphenation')
@option2('--fix-accents/--no-fix-accents')
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
@option2('--n-topics')
@option2('--engine', default="gensim_lda-multicore")
@option2('--passes')
@option2('--alpha')
@option2('--random-seed')
@option2('--workers')
@option2('--max-iter')
@option2('--chunksize')
@option2('--update-every')
@option2('--store-corpus/--no-store-corpus')
@option2('--store-compressed/--no-store-compressed')
@option2('--force-checkpoint/--no-force-checkpoint')
@option2('--enable-checkpoint/--no-enable-checkpoint')
@option2('--passthrough-column')
def click_main(
    options_filename: Optional[str] = None,
    config_filename: str = None,
    corpus_source: Optional[str] = None,
    target_name: str = None,
    train_corpus_folder: Optional[str] = None,
    target_folder: Optional[str] = None,
    fix_hyphenation: bool = True,
    fix_accents: bool = True,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    to_lower: bool = True,
    remove_stopwords: Optional[str] = None,
    min_word_length: int = 2,
    max_word_length: Optional[int] = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    n_topics: int = 50,
    engine: str = "gensim_lda-multicore",
    passes: Optional[int] = None,
    random_seed: Optional[int] = None,
    alpha: str = 'asymmetric',
    workers: Optional[int] = None,
    max_iter: Optional[int] = None,
    chunk_size: Optional[int] = None,
    update_every: Optional[int] = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    passthrough_column: Optional[str] = None,
):
    arguments: dict = update_arguments_from_options_file(arguments=locals(), filename_key='options_filename')

    if not os.path.isfile(config_filename):
        click.echo(f"error: file {config_filename} not found")
        sys.exit(1)

    if arguments.get('target_name') is None:
        click.echo("error: target_name not specified")
        sys.exit(1)

    _main(**arguments)


def _main(
    config_filename: Optional[str] = None,
    target_name: Optional[str] = None,
    corpus_source: Optional[str] = None,
    train_corpus_folder: Optional[str] = None,
    target_folder: Optional[str] = None,
    fix_hyphenation: bool = True,
    fix_accents: bool = True,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    to_lower: bool = True,
    remove_stopwords: Optional[str] = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    n_topics: int = 50,
    engine: str = "gensim_lda-multicore",
    passes: int = None,
    random_seed: int = None,
    alpha: str = 'asymmetric',
    workers: int = None,
    max_iter: int = None,
    chunk_size: Optional[int] = None,
    update_every: Optional[int] = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    passthrough_column: Optional[str] = None,
):
    """Train a new topic model."""
    config: pipeline.CorpusConfig = load_config(config_filename, corpus_source)

    if passthrough_column is None:

        text_transform_opts: pc.TextTransformOpts = pc.TextTransformOpts()

        if fix_accents:
            text_transform_opts.fix_accents = True

        if fix_hyphenation:
            """Replace default dehyphen function"""
            # fix_hyphens: Callable[[str], str] = (
            #     remove_hyphens_fx(config.text_reader_opts.dehyphen_expr)
            #     if config.text_reader_opts.dehyphen_expr is not None
            #     else remove_hyphens
            # )
            text_transform_opts.fix_hyphenation = False
            text_transform_opts.extra_transforms.append(pc.remove_hyphens)

        transform_opts: pc.TokensTransformOpts = pc.TokensTransformOpts(
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

        extract_opts = pc.ExtractTaggedTokensOpts(
            lemmatize=lemmatize,
            pos_includes=pos_includes,
            pos_excludes=pos_excludes,
            **config.pipeline_payload.tagged_columns_names,
        )

    else:
        extract_opts: str = passthrough_column
        text_transform_opts: pc.TextTransformOpts = None
        transform_opts: pc.TokensTransformOpts = None

    engine_args: dict = remove_none(
        dict(
            n_topics=n_topics,
            passes=passes,
            random_seed=random_seed,
            alpha=alpha,
            workers=workers,
            max_iter=max_iter,
            work_folder=os.path.join(target_folder, target_name),
            chunk_size=None,
            update_every=None,
        )
    )

    if corpus_source is None and config.pipeline_payload.source is None:
        click.echo("usage: corpus source must be specified")
        sys.exit(1)

    if not config.pipeline_key_exists("topic_modeling_pipeline"):
        click.echo("config error: `topic_modeling_pipeline` not specified")
        sys.exit(1)

    _: dict = config.get_pipeline(
        pipeline_key="topic_modeling_pipeline",
        config=config,
        target_name=target_name,
        corpus_source=corpus_source,
        train_corpus_folder=train_corpus_folder,
        target_folder=target_folder,
        text_transform_opts=text_transform_opts,
        extract_opts=extract_opts,
        transform_opts=transform_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
        enable_checkpoint=enable_checkpoint,
        force_checkpoint=force_checkpoint,
    ).value()


if __name__ == '__main__':

    click_main()
