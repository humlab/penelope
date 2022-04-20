import os
import sys
from typing import Literal, Optional

import click
from loguru import logger

import penelope.workflows.tm.train_id as workflow
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
@option2('--filename-pattern')
@option2('--target-mode')
@option2('--target-folder', default=None)
@option2('--train-corpus-folder')
@option2('--trained-model-folder')
@option2('--lemmatize/--no-lemmatize')
@option2('--pos-includes')
@option2('--pos-excludes')
@option2('--to-lower/--no-to-lower', default=False)
# @option2('--remove-stopwords')
# @option2('--min-word-length')
# @option2('--max-word-length')
# @option2('--keep-symbols/--no-keep-symbols')
# @option2('--keep-numerals/--no-keep-numerals')
@option2('--max-tokens')
@option2('--tf-threshold')
@option2('--n-topics')
@option2('--engine', default="gensim_lda-multicore")
@option2('--passes')
@option2('--alpha')
@option2('--random-seed')
@option2('--workers')
@option2('--max-iter')
@option2('--num-top-words')
@option2('--chunk-size')
@option2('--update-every')
@option2('--minimum-probability')
@option2('--per-word-topics')
@option2('--store-corpus/--no-store-corpus')
@option2('--store-compressed/--no-store-compressed')
def click_main(
    options_filename: Optional[str] = None,
    config_filename: Optional[str] = None,
    corpus_source: Optional[str] = None,
    filename_pattern: str = None,
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
    # remove_stopwords: Optional[str] = None,
    # min_word_length: int = 2,
    # max_word_length: Optional[int] = None,
    # keep_symbols: bool = False,
    # keep_numerals: bool = False,
    alpha: str = 'asymmetric',
    chunk_size: int = 2000,
    engine: str = "gensim_lda-multicore",
    max_iter: int = None,
    num_top_words: int = None,
    minimum_probability: float = None,
    n_topics: int = 50,
    passes: int = None,
    per_word_topics: bool = False,
    random_seed: int = None,
    update_every: int = 1,
    workers: int = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
):
    arguments: dict = consolidate_cli_arguments(arguments=locals(), filename_key='options_filename')

    main(**arguments)


def main(
    config_filename: Optional[str] = None,
    corpus_source: Optional[str] = None,
    filename_pattern: str = None,
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
    # remove_stopwords: Optional[str] = None,
    # min_word_length: int = 2,
    # max_word_length: int = None,
    # keep_symbols: bool = False,
    # keep_numerals: bool = False,
    alpha: str = 'asymmetric',
    chunk_size: int = 2000,
    engine: str = "gensim_lda-multicore",
    max_iter: int = None,
    num_top_words: int = None,
    minimum_probability: float = None,
    n_topics: int = 50,
    passes: int = None,
    per_word_topics: bool = False,
    random_seed: int = None,
    update_every: int = 1,
    workers: int = None,
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
            alpha=alpha,
            chunk_size=chunk_size,
            max_iter=max_iter,
            num_top_words=num_top_words,
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
    # _: dict = config.get_pipeline(
    #     pipeline_key="topic_modeling_pipeline",

    value: dict = workflow.compute(
        corpus_config=config,
        corpus_source=corpus_source,
        filename_pattern=filename_pattern,
        train_corpus_folder=train_corpus_folder,
        trained_model_folder=trained_model_folder,
        target_mode=target_mode,
        target_folder=target_folder,
        target_name=target_name,
        extract_opts=extract_opts,
        vectorize_opts=vectorize_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
        # transform_opts=transform_opts,
    )

    logger.info(f"workflow completed: model {value.get('target_name')} stored in {value.get('target_folder')}")


if __name__ == '__main__':

    click_main()

    # from click.testing import CliRunner
    # runner = CliRunner()
    # result = runner.invoke(
    #     click_main,
    #     [
    #         '--options-filename',
    #         'penelope/scripts/sample_opts/opts_tm_predict.yml',
    #         '--filename-pattern',
    #         '**/192[0123]/*.feather',
    #     ],
    # )
    # print(result.output)
