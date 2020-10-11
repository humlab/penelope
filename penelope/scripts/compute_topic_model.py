import os
from os.path import join as jj

import click

import penelope.corpus.readers.text_tokenizer as text_tokenizer
import penelope.corpus.tokenized_corpus as tokenized_corpus
import penelope.topic_modelling as topic_modelling
import penelope.utility.file_utility as file_utility

# pylint: disable=unused-argument, too-many-arguments


@click.command()
@click.argument('name')  # , help='Model name.')
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
@click.option('--prefix', default=None, help='Prefix.')
def compute_topic_model(
    name, n_topics, corpus_folder, corpus_filename, engine, passes, random_seed, alpha, workers, max_iter, prefix
):
    run_model(
        name, n_topics, corpus_folder, corpus_filename, engine, passes, random_seed, alpha, workers, max_iter, prefix
    )


def run_model(
    name, n_topics, corpus_folder, corpus_filename, engine, passes, random_seed, alpha, workers, max_iter, prefix
):
    """ runner """

    call_arguments = dict(locals())

    if corpus_folder is None:
        corpus_folder, _ = os.path.split(os.path.abspath(corpus_filename))

    os.makedirs(jj(corpus_folder, name), exist_ok=True)

    topic_modeling_opts = {
        k: v
        for k, v in call_arguments.items()
        if k in ['n_topics', 'passes', 'random_seed', 'alpha', 'workers', 'max_iter', 'prefix'] and v is not None
    }

    transformer_opts = dict(
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

    # if SparvTokenizer opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)

    tokenizer = text_tokenizer.TextTokenizer(
        source_path=corpus_filename,
        chunk_size=None,
        filename_pattern="*.txt",
        filename_filter=None,
        fix_hyphenation=True,
        fix_whitespaces=False,
        filename_fields=file_utility.filename_field_parser(['year:_:1', 'sequence_id:_:2']),
    )

    corpus = tokenized_corpus.TokenizedCorpus(reader=tokenizer, **transformer_opts)

    train_corpus = topic_modelling.TrainingCorpus(
        terms=corpus,
        doc_term_matrix=None,
        id2word=None,
        documents=corpus.documents,
    )

    inferred_model = topic_modelling.infer_model(
        train_corpus=train_corpus,
        method=engine,
        engine_args=topic_modeling_opts,
    )

    inferred_model.topic_model.save(jj(corpus_folder, name, 'gensim.model'))

    topic_modelling.store_model(inferred_model, jj(corpus_folder, name))

    inferred_topics = topic_modelling.compile_inferred_topics_data(inferred_model.topic_model, train_corpus.corpus, train_corpus.id2word, train_corpus.documents)
    inferred_topics.store(corpus_folder, name)


if __name__ == '__main__':
    compute_topic_model()  # pylint: disable=no-value-for-parameter
