import os
import sys
from typing import Optional

import click
from penelope.pipeline import CorpusPipeline
from penelope.pipeline.topic_model.pipelines import from_id_tagged_frame_pipeline
from penelope.scripts.utils import load_config, remove_none, update_arguments_from_options_file

# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('config-filename', required=False)
@click.argument('target-name', required=False)
@click.option('--options-filename', default=None, help='Use values in YAML file as command line options.')
@click.option('--corpus-source', default=None, help='Corpus filename/folder (overrides config)')
@click.option('--target-folder', default=None, help='Target folder, if none then corpus-folder/target-name.')
@click.option('--train-corpus-folder', default=None, type=click.STRING, help='Use train corpus in folder if exists')
@click.option('--lemmatize/--no-lemmatize', default=True, is_flag=True, help='Use word baseforms')
# @click.option('--pos-includes', default='', help='POS tags to include e.g. "|NN|JJ|".', type=click.STRING)
# @click.option('--pos-excludes', default='', help='POS tags to exclude e.g. "|MAD|MID|PAD|".', type=click.STRING)
# @click.option('--to-lower/--no-to-lower', default=True, is_flag=True, help='Lowercase words')
# @click.option('--min-word-length', default=1, type=click.IntRange(1, 99), help='Min length of words to keep')
# @click.option('--max-word-length', default=None, type=click.IntRange(10, 99), help='Max length of words to keep')
# @click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
# @click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
# @click.option('--remove-stopwords', default=None, type=click.Choice(['swedish', 'english']), help='Remove stopwords')
@click.option('--n-topics', default=50, help='Number of topics.', type=click.INT)
@click.option('--engine', default="gensim_lda-multicore", help='LDA implementation')
@click.option('--passes', default=None, help='Number of passes.', type=click.INT)
@click.option('--alpha', default='asymmetric', help='Prior belief of topic probability. symmetric/asymmertic/auto')
@click.option('--random-seed', default=None, help="Random seed value", type=click.INT)
@click.option('--workers', default=None, help='Number of workers (if applicable).', type=click.INT)
@click.option('--max-iter', default=None, help='Max number of iterations.', type=click.INT)
@click.option('--store-corpus/--no-store-corpus', default=True, is_flag=True, help='')
@click.option('--store-compressed/--no-store-compressed', default=True, is_flag=True, help='')
def typer_main(
    options_filename: Optional[str] = None,
    target_name: Optional[str] = None,
    corpus_source: Optional[str] = None,
    config_filename: Optional[str] = None,
    train_corpus_folder: Optional[str] = None,
    target_folder: Optional[str] = None,
    lemmatize: bool = True,
    # pos_includes: str = '',
    # pos_excludes: str = '',
    # remove_stopwords: Optional[str] = None,
    # min_word_length: int = 2,
    # max_word_length: int = None,
    # keep_symbols: bool = False,
    # keep_numerals: bool = False,
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
    # pos_includes: str = '',
    # pos_excludes: str = '',
    # remove_stopwords: Optional[str] = None,
    # min_word_length: int = 2,
    # max_word_length: int = None,
    # keep_symbols: bool = False,
    # keep_numerals: bool = False,
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

    # transform_opts: pc.TokensTransformOpts = pc.TokensTransformOpts(
    #     to_lower=to_lower,
    #     to_upper=False,
    #     min_len=min_word_length,
    #     max_len=max_word_length,
    #     remove_accents=False,
    #     remove_stopwords=(remove_stopwords is not None),
    #     stopwords=None,
    #     extra_stopwords=None,
    #     language=remove_stopwords,
    #     keep_numerals=keep_numerals,
    #     keep_symbols=keep_symbols,
    #     only_alphabetic=only_alphabetic,
    #     only_any_alphanumeric=only_any_alphanumeric,
    # )

    # extract_opts = pc.ExtractTaggedTokensOpts(
    #     lemmatize=lemmatize,
    #     pos_includes=pos_includes,
    #     pos_excludes=pos_excludes,
    #     **config.pipeline_payload.tagged_columns_names,
    # )

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
    tagged_column: str = 'lemma_id' if lemmatize else 'token_id'
    tm_pipeline: CorpusPipeline = from_id_tagged_frame_pipeline
    # _: dict = config.get_pipeline(
    #     pipeline_key="topic_modeling_pipeline",

    _: dict = tm_pipeline(
        corpus_config=config,
        target_name=target_name,
        corpus_source=corpus_source,
        tagged_column=tagged_column,
        train_corpus_folder=train_corpus_folder,
        target_folder=target_folder,
        # extract_opts=extract_opts,
        # transform_opts=transform_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
    ).value()


if __name__ == '__main__':

    typer_main()
