import os
import sys
from os.path import dirname, isdir, isfile
from typing import Optional

import click
import penelope.corpus as pc
import yaml
from penelope import pipeline
from penelope.utility import PropertyValueMaskingOpts

# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('config-filename', required=True)
@click.argument('target-name', required=False)
@click.option('--options-filename', default=None, help='Use values in YAML file as command line options.')
@click.option('--corpus-source', default=None, help='Corpus filename/folder (overrides config)')
@click.option('--target-folder', default=None, help='Target folder, if none then corpus-folder/target-name.')
@click.option('--train-corpus-folder', default=None, type=click.STRING, help='Use train corpus in folder if exists')
@click.option('--fix-hyphenation/--no-fix-hyphenation', default=True, is_flag=True, help='Fix hyphens')
@click.option('--fix-accents/--no-fix-accents', default=True, is_flag=True, help='Fix accents')
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
@click.option('--n-topics', default=50, help='Number of topics.', type=click.INT)
@click.option('--engine', default="gensim_lda-multicore", help='LDA implementation')
@click.option('--passes', default=None, help='Number of passes.', type=click.INT)
@click.option('--alpha', default='asymmetric', help='Prior belief of topic probability. symmetric/asymmertic/auto')
@click.option('--random-seed', default=None, help="Random seed value", type=click.INT)
@click.option('--workers', default=None, help='Number of workers (if applicable).', type=click.INT)
@click.option('--max-iter', default=None, help='Max number of iterations.', type=click.INT)
@click.option('--store-corpus/--no-store-corpus', default=True, is_flag=True, help='')
@click.option('--store-compressed/--no-store-compressed', default=True, is_flag=True, help='')
@click.option('--force-checkpoint/--no-force-checkpoint', default=False, is_flag=True, help='')
@click.option('--enable-checkpoint/--no-enable-checkpoint', default=True, is_flag=True, help='')
@click.option('--passthrough-column', default=None, type=click.STRING, help="Use tagged columns as-is (ignore filters)")
def click_main(
    config_filename: str = None,
    target_name: str = None,
    options_filename: Optional[str] = None,
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
    arguments: dict = locals()
    print(arguments)
    del arguments['options_filename']

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
    """Create a topic model.

    Args:
        config_filename (Optional[str], optional): [description]. Defaults to None.
        target_name (Optional[str], optional): [description]. Defaults to None.
        corpus_source (Optional[str], optional): [description]. Defaults to None.
        train_corpus_folder (Optional[str], optional): [description]. Defaults to None.
        target_folder (Optional[str], optional): [description]. Defaults to None.
        fix_hyphenation (bool, optional): [description]. Defaults to True.
        fix_accents (bool, optional): [description]. Defaults to True.
        lemmatize (bool, optional): [description]. Defaults to True.
        pos_includes (str, optional): [description]. Defaults to ''.
        pos_excludes (str, optional): [description]. Defaults to ''.
        to_lower (bool, optional): [description]. Defaults to True.
        remove_stopwords (Optional[str], optional): [description]. Defaults to None.
        min_word_length (int, optional): [description]. Defaults to 2.
        max_word_length (int, optional): [description]. Defaults to None.
        keep_symbols (bool, optional): [description]. Defaults to False.
        keep_numerals (bool, optional): [description]. Defaults to False.
        only_any_alphanumeric (bool, optional): [description]. Defaults to False.
        only_alphabetic (bool, optional): [description]. Defaults to False.
        n_topics (int, optional): [description]. Defaults to 50.
        engine (str, optional): [description]. Defaults to "gensim_lda-multicore".
        passes (int, optional): [description]. Defaults to None.
        random_seed (int, optional): [description]. Defaults to None.
        alpha (str, optional): [description]. Defaults to 'asymmetric'.
        workers (int, optional): [description]. Defaults to None.
        max_iter (int, optional): [description]. Defaults to None.
        store_corpus (bool, optional): [description]. Defaults to True.
        store_compressed (bool, optional): [description]. Defaults to True.
        enable_checkpoint (bool, optional): [description]. Defaults to True.
        force_checkpoint (bool, optional): [description]. Defaults to False.
        passthrough_column (Optional[str], optional): [description]. Defaults to None.
    """
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(path=config_filename)
    if config.pipeline_payload.source is None:
        config.pipeline_payload.source = corpus_source
        if isdir(corpus_source):
            config.folders(corpus_source, method='replace')
        elif isfile(corpus_source):
            config.folders(dirname(corpus_source), method='replace')

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

    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts()

    engine_args = {
        k: v
        for k, v in {
            'n_topics': n_topics,
            'passes': passes,
            'random_seed': random_seed,
            'alpha': alpha,
            'workers': workers,
            'max_iter': max_iter,
            'work_folder': os.path.join(target_folder, target_name),
        }.items()
        if v is not None
    }

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
        filter_opts=filter_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
        enable_checkpoint=enable_checkpoint,
        force_checkpoint=force_checkpoint,
    ).value()


RUN_MODE = "production"

if __name__ == '__main__':

    if RUN_MODE == "production":

        click_main()

    # else:
    #     logger.warning("RUNNING IN DEBUG MODE")

    #     from click.testing import CliRunner

    #     runner = CliRunner()
    #     result = runner.invoke(
    #         click_main,
    #         [
    #             '--n-topics',
    #             '200',
    #             # '--lemmatize',
    #             # '--to-lower',
    #             # '--min-word-length',
    #             1,
    #             '--only-any-alphanumeric',
    #             '--engine',
    #             'gensim_lda-multicore',
    #             '--random-seed',
    #             42,
    #             '--alpha',
    #             'asymmetric',
    #             '--max-iter',
    #             3000,
    #             '--store-corpus',
    #             '--workers',
    #             6,
    #             '--target-folder',
    #             '/home/roger/source/penelope/data',
    #             '/home/roger/source/penelope/riksprot-parlaclarin.yml',
    #             'riksprot-parlaclarin-protokoll-200-lemma',
    #             1,
    #         ],
    #     )
    #     print(result.output)
