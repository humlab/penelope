import os
import sys
from os.path import join as jj
from typing import Optional

from penelope.corpus import TextReaderOpts, TextTransformOpts, TokensTransformOpts
from penelope.scripts.utils import option2
from penelope.workflows.tm import train_legacy as workflow

try:
    import click
except ImportError:
    ...


# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('target-name')  # , help='Model name.')
@option2('--corpus-source', default=None)
@option2('--corpus-folder', default=None)
@option2('--n-topics', default=50, type=click.INT)
@option2('--engine', default="gensim_lda-multicore")
@option2('--passes', default=None, type=click.INT)
@option2('--alpha', default='asymmetric')
@option2('--random-seed', default=None, type=click.INT)
@option2('--workers', default=None, type=click.INT)
@option2('--max-iter', default=None, type=click.INT)
@option2('--store-corpus/--no-store-corpus', default=True, is_flag=True)
@option2('--store-compressed/--no-store-compressed', default=True, is_flag=True)
@option2('--n-tokens', default=200, type=click.INT)
@option2('--filename-field', '-f', default=None, multiple=True)
@click.option('--work-folder', default=None, help='Work folder (MALLET `prefix`).')
@click.option('--minimum-probability', default=0.001, help='minimum-probability.', type=click.FLOAT)
def click_main(
    target_name,
    corpus_folder,
    corpus_source: Optional[str] = None,
    n_topics: int = 50,
    n_tokens: int = 200,
    engine: str = "gensim_lda-multicore",
    passes: int = None,
    random_seed: int = None,
    alpha: str = 'asymmetric',
    workers: int = None,
    max_iter: int = None,
    work_folder: str = None,
    filename_field=None,
    store_corpus: bool = True,
    store_compressed: bool = True,
    minimum_probability: float = 0.001,
):

    topic_modeling_opts = {
        k: v
        for k, v in {
            'n_topics': n_topics,
            'passes': passes,
            'random_seed': random_seed,
            'alpha': alpha,
            'workers': workers,
            'max_iter': max_iter,
            'work_folder': work_folder,
        }.items()
        if v is not None
    }

    main(
        target_name=target_name,
        corpus_folder=corpus_folder,
        corpus_source=corpus_source,
        engine=engine,
        engine_args=topic_modeling_opts,
        filename_field=filename_field,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
        n_tokens=n_tokens,
        minimum_probability=minimum_probability,
    )


def main(
    target_name: str = None,
    corpus_folder: str = None,
    corpus_source: str = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    filename_pattern="*.txt",
    filename_field: str = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    n_tokens: int = 200,
    minimum_probability: float = 0.001,
):
    """runner"""

    if corpus_source is None and corpus_folder is None:
        click.echo("usage: either corpus-folder or corpus filename must be specified")
        sys.exit(1)

    if len(filename_field or []) == 0:
        click.echo("warning: no filename metadata fields specified (use option --filename-field)")
        # sys.exit(1)

    if corpus_folder is None:
        corpus_folder, _ = os.path.split(os.path.abspath(corpus_source))

    target_folder: str = jj(corpus_folder, target_name)

    os.makedirs(target_folder, exist_ok=True)

    transform_opts: TokensTransformOpts = TokensTransformOpts(
        only_alphabetic=False,
        only_any_alphanumeric=True,
        to_lower=True,
        min_len=2,
        max_len=99,
        remove_accents=False,
        remove_stopwords=True,
        stopwords=None,
        extra_stopwords=None,
        language="swedish",
        keep_numerals=False,
        keep_symbols=False,
    )

    reader_opts: TextReaderOpts = TextReaderOpts(
        filename_pattern=filename_pattern,
        filename_filter=None,
        filename_fields=filename_field,
    )

    text_transform_opts: TextTransformOpts = TextTransformOpts(fix_whitespaces=False, fix_hyphenation=True)

    _ = workflow.compute(
        target_name=target_name,
        corpus_source=corpus_source,
        target_folder=target_folder,
        reader_opts=reader_opts,
        text_transform_opts=text_transform_opts,
        transform_opts=transform_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
        n_tokens=n_tokens,
        minimum_probability=minimum_probability,
    )


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
