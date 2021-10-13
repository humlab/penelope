import os
import sys
from os.path import join as jj

import click
import penelope.topic_modelling as tm
from penelope.corpus import TextReaderOpts, TextTransformOpts, TokenizedCorpus, TokensTransformOpts
from penelope.corpus.readers import TextTokenizer

# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('target-name')  # , help='Model name.')
@click.option('--n-topics', default=50, help='Number of topics.', type=click.INT)
@click.option('--corpus-folder', default=None, help='Corpus folder (if vectorized corpus exists on disk).')
@click.option(
    '--corpus-filename',
    help='Corpus filename (if text corpus file or folder, or Sparv XML). Corpus tag if vectorized corpus.',
)
@click.option('--engine', default="gensim_lda-multicore", help='LDA implementation')
@click.option('--passes', default=None, help='Number of passes.', type=click.INT)
@click.option('--alpha', default='asymmetric', help='Prior belief of topic probability. symmetric/asymmertic/auto')
@click.option('--random-seed', default=None, help="Random seed value", type=click.INT)
@click.option('--workers', default=None, help='Number of workers (if applicable).', type=click.INT)
@click.option('--max-iter', default=None, help='Max number of iterations.', type=click.INT)
@click.option('--work-folder', default=None, help='Work folder (MALLET `prefix`).')
@click.option('--filename-field', '-f', default=None, help='Field to extract from document name', multiple=True)
@click.option('--store-corpus/--no-store-corpus', default=True, is_flag=True, help='')
@click.option('--compressed/--no-compressed', default=True, is_flag=True, help='')
@click.option('--n-tokens', default=200, help='Number tokens per topic.', type=click.INT)
@click.option('--minimum-probability', default=0.001, help='minimum-probability.', type=click.FLOAT)
def click_main(
    target_name,
    n_topics,
    corpus_folder,
    corpus_filename,
    engine,
    passes,
    random_seed,
    alpha,
    workers: int,
    max_iter: int,
    work_folder: str,
    filename_field,
    store_corpus: bool = True,
    compressed: bool = True,
    n_tokens: int = 200,
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
        corpus_filename=corpus_filename,
        engine=engine,
        engine_args=topic_modeling_opts,
        filename_field=filename_field,
        store_corpus=store_corpus,
        store_compressed=compressed,
        n_tokens=n_tokens,
        minimum_probability=minimum_probability,
    )


def main(
    target_name: str = None,
    corpus_folder: str = None,
    corpus_filename: str = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    filename_pattern="*.txt",
    filename_field: str = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    n_tokens: int = 200,
    minimum_probability: float = 0.001,
):
    """ runner """

    if corpus_filename is None and corpus_folder is None:
        click.echo("usage: either corpus-folder or corpus filename must be specified")
        sys.exit(1)

    if len(filename_field or []) == 0:
        click.echo("warning: no filename metadata fields specified (use option --filename-field)")
        # sys.exit(1)

    if corpus_folder is None:
        corpus_folder, _ = os.path.split(os.path.abspath(corpus_filename))

    target_folder: str = jj(corpus_folder, target_name)

    os.makedirs(target_folder, exist_ok=True)

    transformer_opts: TokensTransformOpts = TokensTransformOpts(
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

    transform_opts: TextTransformOpts = TextTransformOpts(fix_whitespaces=False, fix_hyphenation=True)

    tokens_reader = TextTokenizer(
        source=corpus_filename,
        transform_opts=transform_opts,
        reader_opts=reader_opts,
        chunk_size=None,
    )

    corpus: TokenizedCorpus = TokenizedCorpus(reader=tokens_reader, transform_opts=transformer_opts)

    train_corpus: tm.TrainingCorpus = tm.TrainingCorpus(
        terms=corpus.terms,
        doc_term_matrix=None,
        id2token=None,
        document_index=corpus.document_index,
        corpus_options=dict(
            reader_opts=reader_opts.props,
            transform_opts=transformer_opts.props,
        ),
    )

    inferred_model: tm.InferredModel = tm.train_model(
        train_corpus=train_corpus,
        method=engine,
        engine_args=engine_args,
    )

    inferred_model.topic_model.save(jj(target_folder, 'gensim.model.gz'))

    inferred_model.store(target_folder, store_corpus=store_corpus, store_compressed=store_compressed)

    inferred_topics: tm.InferredTopicsData = tm.predict_topics(
        inferred_model.topic_model,
        corpus=train_corpus.corpus,
        id2token=train_corpus.id2token,
        document_index=train_corpus.document_index,
        n_tokens=n_tokens,
        minimum_probability=minimum_probability,
    )

    inferred_topics.store(target_folder)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
