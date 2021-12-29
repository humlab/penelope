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
@option2('--train-corpus-folder', default=None, type=click.STRING)
@option2('--fix-hyphenation/--no-fix-hyphenation', default=True, is_flag=True)
@option2('--fix-accents/--no-fix-accents', default=True, is_flag=True)
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
@option2('--n-topics', default=50, type=click.INT)
@option2('--engine', default="gensim_lda-multicore")
@option2('--passes', default=None, type=click.INT)
@option2('--alpha', default='asymmetric')
@option2('--random-seed', default=None, type=click.INT)
@option2('--workers', default=None, type=click.INT)
@option2('--max-iter', default=None, type=click.INT)
@option2('--store-corpus/--no-store-corpus', default=True, is_flag=True)
@option2('--store-compressed/--no-store-compressed', default=True, is_flag=True)
@option2('--force-checkpoint/--no-force-checkpoint', default=False, is_flag=True)
@option2('--enable-checkpoint/--no-enable-checkpoint', default=True, is_flag=True)
@option2('--passthrough-column', default=None, type=click.STRING)
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
