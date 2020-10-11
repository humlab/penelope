from penelope.topic_modelling.container import TrainingCorpus
from tests.test_data.tranströmer_corpus import TranströmerCorpus
import penelope.topic_modelling as topic_modelling


def test_tranströmers_corpus():

    corpus = TranströmerCorpus()
    for filename, tokens in corpus:
        assert len(filename) > 0
        assert len(tokens) > 0


def test_compute_model():

    # engine = "gensim_mallet-lda"
    engine = "gensim_lda"

    topic_modeling_opts = {
        'n_topics': 4,
        'passes': 1,
        'random_seed': 42,
        'alpha': 'auto',
        'workers': 1,
        'max_iter': 100,
        'prefix': '',
    }

    corpus = TranströmerCorpus()

    train_corpus = TrainingCorpus(
        terms=corpus.terms,
        documents=corpus.documents,
    )

    inferred_model, inferred_topics = topic_modelling.infer_model(
        train_corpus=train_corpus,
        method=engine,
        engine_args=topic_modeling_opts,
    )

    assert inferred_topics is not None
    assert inferred_model is not None

    # if corpus_folder is None:
    #     corpus_folder, _ = os.path.split(os.path.abspath(corpus_filename))

    # os.makedirs(jj(corpus_folder, name), exist_ok=True)

    # model_data.topic_model.save(jj(corpus_folder, name, 'gensim.model'))

    # topic_modelling.store_model(model_data, jj(corpus_folder, name))

    # corpus_data.store(corpus_folder, name)
