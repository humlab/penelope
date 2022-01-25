import os
import sys
from typing import Literal, Optional

import click

from penelope import corpus as pc
from penelope import pipeline
from penelope.scripts.utils import consolidate_cli_arguments, load_config, option2, remove_none
from penelope.topic_modelling.interfaces import InferredModel

# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('config-filename', required=False)
@click.argument('target-name', required=False)
@option2('--options-filename')
@option2('--corpus-source', default=None)
@option2('--target-mode')
@option2('--target-folder', default=None)
@option2('--train-corpus-folder')
@option2('--trained-model-folder')
@option2('--lemmatize/--no-lemmatize')
@option2('--pos-includes')
@option2('--pos-excludes')
@option2('--to-lower/--no-to-lower')
@option2('--fix-hyphenation/--no-fix-hyphenation')
@option2('--fix-accents/--no-fix-accents')
@option2('--remove-stopwords')
@option2('--min-word-length')
@option2('--max-word-length')
@option2('--keep-symbols/--no-keep-symbols')
@option2('--keep-numerals/--no-keep-numerals')
@option2('--only-alphabetic')
@option2('--only-any-alphanumeric')
@option2('--max-tokens')
@option2('--tf-threshold')
@option2('--n-topics')
@option2('--engine', default="gensim_lda-multicore")
@option2('--passes')
@option2('--alpha')
@option2('--random-seed')
@option2('--workers')
@option2('--max-iter')
@option2('--chunk-size')
@option2('--update-every')
@option2('--minimum-probability')
@option2('--per-word-topics')
@option2('--store-corpus/--no-store-corpus')
@option2('--store-compressed/--no-store-compressed')
@option2('--force-checkpoint/--no-force-checkpoint')
@option2('--enable-checkpoint/--no-enable-checkpoint')
@option2('--passthrough-column')
def click_main(
    options_filename: Optional[str] = None,
    config_filename: Optional[str] = None,
    corpus_source: Optional[str] = None,
    train_corpus_folder: Optional[str] = None,
    trained_model_folder: Optional[str] = None,
    target_mode: Literal['train', 'predict', 'both'] = 'both',
    target_folder: Optional[str] = None,
    target_name: Optional[str] = None,
    to_lower: bool = True,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    max_tokens: int = None,
    tf_threshold: int = None,
    remove_stopwords: Optional[str] = None,
    min_word_length: int = 2,
    max_word_length: Optional[int] = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    alpha: str = 'asymmetric',
    chunk_size: int = 2000,
    engine: str = "gensim_lda-multicore",
    max_iter: int = None,
    minimum_probability: float = None,
    n_topics: int = 50,
    passes: int = None,
    per_word_topics: bool = False,
    random_seed: int = None,
    update_every: int = 1,
    workers: int = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
    fix_hyphenation: bool = True,
    fix_accents: bool = True,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    passthrough_column: Optional[str] = None,
):
    arguments: dict = consolidate_cli_arguments(arguments=locals(), filename_key='options_filename')

    main(**arguments)


def main(
    config_filename: Optional[str] = None,
    corpus_source: Optional[str] = None,
    train_corpus_folder: Optional[str] = None,
    trained_model_folder: Optional[str] = None,
    target_mode: Literal['train', 'predict', 'both'] = 'both',
    target_folder: Optional[str] = None,
    target_name: Optional[str] = None,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    to_lower: bool = True,
    max_tokens: int = None,
    tf_threshold: int = None,
    remove_stopwords: Optional[str] = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    alpha: str = 'asymmetric',
    chunk_size: int = 2000,
    engine: str = "gensim_lda-multicore",
    max_iter: int = None,
    minimum_probability: float = None,
    n_topics: int = 50,
    passes: int = None,
    per_word_topics: bool = False,
    random_seed: int = None,
    update_every: int = 1,
    workers: int = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    fix_hyphenation: bool = True,
    fix_accents: bool = True,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
    passthrough_column: Optional[str] = None,
):

    if not config_filename or not os.path.isfile(config_filename):
        click.echo("error: config file not specified/found")
        raise sys.exit(1)

    if target_name is None:
        click.echo("error: target_name not specified")
        raise sys.exit(1)

    if target_mode == 'predict' and not InferredModel.exists(trained_model_folder):
        click.echo("error: trained model folder not specified")
        raise sys.exit(1)

    config: pipeline.CorpusConfig = load_config(config_filename, corpus_source)

    if corpus_source is None and config.pipeline_payload.source is None:
        click.echo("usage: corpus source must be specified")
        sys.exit(1)

    if not config.pipeline_key_exists("topic_modeling_pipeline"):
        click.echo("config error: `topic_modeling_pipeline` not specified")
        sys.exit(1)

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

    extract_opts: pc.ExtractTaggedTokensOpts = pc.ExtractTaggedTokensOpts(
        lemmatize=lemmatize,
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        **config.pipeline_payload.tagged_columns_names,
    )

    if passthrough_column is not None:

        extract_opts: str = passthrough_column
        text_transform_opts: pc.TextTransformOpts = None
        transform_opts: pc.TokensTransformOpts = None

    engine_args: dict = remove_none(
        dict(
            alpha=alpha,
            chunk_size=chunk_size,
            max_iter=max_iter,
            minimum_probability=minimum_probability,
            n_topics=n_topics,
            passes=passes,
            per_word_topics=per_word_topics,
            random_seed=random_seed,
            update_every=update_every,
            work_folder=os.path.join(target_folder, target_name),
            workers=workers,
        )
    )

    _: dict = config.get_pipeline(
        pipeline_key="topic_modeling_pipeline",
        config=config,
        corpus_source=corpus_source,
        train_corpus_folder=train_corpus_folder,
        trained_model_folder=trained_model_folder,
        target_mode=target_mode,
        target_folder=target_folder,
        target_name=target_name,
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
