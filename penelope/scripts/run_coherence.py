import pickle
import types

import click

# import penelope.topic_modelling as topic_modelling

# pylint: disable=too-many-arguments
ENGINE_OPTIONS = [
    ('MALLET LDA', 'gensim_mallet-lda'),
    ('gensim LDA', 'gensim_lda'),
    ('gensim LSI', 'gensim_lsi'),
    ('gensim HDP', 'gensim_hdp'),
    ('gensim DTM', 'gensim_dtm'),
    ('scikit LDA', 'sklearn_lda'),
    ('scikit NMF', 'sklearn_nmf'),
    ('scikit LSA', 'sklearn_lsa'),
    ('STTM   LDA', 'gensim_sttm-lda'),
    ('STTM   BTM', 'gensim_sttm-btm'),
    ('STTM   PTM', 'gensim_sttm-ptm'),
    ('STTM  SATM', 'gensim_sttm-satm'),
    ('STTM   DMM', 'gensim_sttm-dmm'),
    ('STTM  WATM', 'gensim_sttm-watm'),
]


def store_model(data, filename):

    data = types.SimpleNamespace(
        topic_model=data.topic_model,
        id2term=data.id2term,
        bow_corpus=data.bow_corpus,
        doc_term_matrix=None,  # doc_term_matrix,
        doc_topic_matrix=None,  # doc_topic_matrix,
        vectorizer=None,  # vectorizer,
        processed=data.processed,
        coherence_scores=data.coherence_scores,
    )

    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


@click.command()
@click.option('--n-start', default=50, help='Number of topics, start.')
@click.option('--n-stop', default=250, help='Number of topics, stop.')
@click.option('--n-step', default=25, help='Number of topics, step.')
@click.option('--data-folder', default='./', help='Corpus folder.')
@click.option('--engine', default="gensim_lda-multicore", help='LDA implementation')
@click.option('--passes', default=None, help='Number of passes.')
@click.option('--alpha', default='symmetric', help='Prior belief of topic probability.')
@click.option('--workers', default=None, help='Number of workers (if applicable).')
@click.option('--prefix', default=None, help='Prefix.')
def compute(n_start, n_stop, n_step, data_folder, engine, passes, alpha, workers, prefix):
    """ runner """
    raise NotImplementedError("This script is specfic to political case - needs to be adapted")
    # if engine not in [y for x, y in ENGINE_OPTIONS]:
    #     logging.error("Unknown method {}".format(engine))

    # # dtm, document_index, id2token = corpus_data.load_as_dtm2(data_folder, [1, 3])

    # kwargs = dict(n_start=n_start, n_stop=n_stop, n_step=n_step)

    # if workers is not None:
    #     kwargs.update(dict(workers=workers))

    # if passes is not None:
    #     kwargs.update(dict(passes=passes))

    # if prefix is not None:
    #     kwargs.update(dict(prefix=prefix))

    # _, inferred_topics = topic_modelling.compute_model(
    #     doc_term_matrix=dtm, id2word=id2token, document_index=document_index, method=engine, engine_args=kwargs
    # )


if __name__ == '__main__':
    compute()  # pylint: disable=no-value-for-parameter
