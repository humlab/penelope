import gensim

from .wrappers import MalletTopicModel, STTMTopicModel

TEMP_PATH = './tmp/'

# OBS OBS! https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
# DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)

# default_options = {
#     'LSI': {'engine': gensim.models.LsiModel, 'options': {'corpus': None, 'num_topics': 20, 'id2word': None}}
# }


# pylint: disable=too-many-return-statements, inconsistent-return-statements
def engine_options(algorithm: str, corpus, id2word, kwargs) -> dict:

    if algorithm == 'LSI':
        return {
            'engine': gensim.models.LsiModel,
            'options': {
                'corpus': corpus,
                'num_topics': kwargs.get('n_topics', 0),
                'id2word': id2word,
                'power_iters': 2,
                'onepass': True,
            },
        }

    if algorithm == 'LDA':
        return {
            'engine': gensim.models.LdaModel,
            'options': {
                # distributed=False, chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None, minimum_phi_value=0.01, per_word_topics=False, callbacks=None, dtype=<class 'numpy.float32'>)¶
                'corpus': corpus,
                'num_topics': int(kwargs.get('n_topics', 20)),
                'id2word': id2word,
                'iterations': kwargs.get('max_iter', 1000),
                'passes': int(kwargs.get('passes', 1)),
                'eval_every': 2,
                'update_every': 10,  # Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning.
                'alpha': 'auto',
                'eta': 'auto',  # None
                # 'decay': 0.1, # 0.5
                # 'chunksize': int(kwargs.get('chunksize', 1)),
                'per_word_topics': True,
                # 'random_state': 100
                # 'offset': 1.0,
                # 'dtype': np.float64
                # 'callbacks': [
                #    gensim.models.callbacks.PerplexityMetric(corpus=corpus, logger='visdom'),
                #    gensim.models.callbacks.ConvergenceMetric(distance='jaccard', num_words=100, logger='shell')
                # ]
            },
        }

    if algorithm == 'LDA-MULTICORE':
        return {
            'engine': gensim.models.LdaMulticore,
            'options': {
                # workers=None, chunksize=2000, passes=1, batch=False, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, random_state=None, minimum_probability=0.01, minimum_phi_value=0.01, per_word_topics=False, dtype=<class 'numpy.float32'>)v
                'corpus': corpus,  # Sream of document vectors or sparse matrix of shape (num_terms, num_documents).
                'num_topics': int(kwargs.get('n_topics', 20)),
                'id2word': id2word,  # id2word ({dict of (int, str), gensim.corpora.dictionary.Dictionary})
                'iterations': kwargs.get(
                    'max_iter', 2000
                ),  # Maximum number of iterations through the corpus when inferring the topic distribution of a corpus
                'passes': int(kwargs.get('passes', 1)),  # Number of passes through the corpus during training.
                'workers': kwargs.get(
                    'workers', 2
                ),  # set workers directly to the number of your real cores (not hyperthreads) minus one
                'eta': 'auto',  # A-priori belief on word probability
                'per_word_topics': True,
                # 'random_state': 100                            # Either a randomState object or a seed to generate one. Useful for reproducibility.
                # 'decay': 0.5,                                  # Kappa from Matthew D. Hoffman, David M. Blei, Francis Bach:
                # 'chunksize': 2000,                             # chunksize (int, optional) – Number of documents to be used in each training chunk.
                # 'eval_every': 10                               # Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
                # 'offset': 1.0,                                 # Tau_0 from Matthew D. Hoffman, David M. Blei, Francis Bach
                # 'dtype': np.float64
                # 'callbacks': [
                #    gensim.models.callbacks.PerplexityMetric(corpus=corpus, logger='visdom'),
                #    gensim.models.callbacks.ConvergenceMetric(distance='jaccard', num_words=100, logger='shell')
                # ]
                # gamma_threshold                               # Minimum change in the value of the gamma parameters to continue iterating.
                # minimum_probability                           # Topics with a probability lower than this threshold will be filtered out.
                # minimum_phi_value                             # if per_word_topics is True, this represents a lower bound on the term probabilities.
                # per_word_topics                               # If True, the model also computes a list of topics, sorted in descending order of most likely topics
                # dtype                                         # Data-type to use during calculations inside model.
            },
        }

    if algorithm == 'HDP':
        return {
            'engine': gensim.models.HdpModel,
            'options': {
                'corpus': corpus,
                'T': kwargs.get('n_topics', 0),
                'id2word': id2word,
                # 'iterations': kwargs.get('max_iter', 0),
                # 'passes': kwargs.get('passes', 20),
                # 'alpha': 'auto'
            },
        }

    if algorithm == 'DTM':
        # Note, mandatory: 'time_slice': textacy_utility.count_documents_in_index_by_pivot(documents, year_column)
        return {
            'engine': gensim.models.LdaSeqModel,
            'options': {
                'corpus': corpus,
                'num_topics': kwargs.get('n_topics', 0),
                'id2word': id2word,
                # 'time_slice': textacy_utility.count_documents_in_index_by_pivot(documents, year_column),
                # 'initialize': 'gensim/own/ldamodel',
                # 'lda_model': model # if initialize='gensim'
                # 'lda_inference_max_iter': kwargs.get('max_iter', 0),
                # 'passes': kwargs.get('passes', 20),
                # 'alpha': 'auto'
            },
        }

    if algorithm == 'MALLET-LDA':
        return {
            # num_topics=100, alpha=50, id2word=None, workers=4, prefix=None, optimize_interval=0, iterations=2000, topic_threshold=0.0, random_seed=0)¶
            'engine': MalletTopicModel,
            'options': {
                'corpus': corpus,  # Collection of texts in BoW format.
                'id2word': id2word,  # Dictionary
                'default_mallet_home': '/usr/local/share/mallet-2.0.8/',  # MALLET_HOME
                'num_topics': kwargs.get('n_topics', 100),  # Number of topics.
                'iterations': kwargs.get('max_iter', 3000),  # Number of training iterations.
                # 'alpha': int(kwargs.get('passes', 20))                     # Alpha parameter of LDA.
                'prefix': kwargs.get('prefix', TEMP_PATH),
                'workers': int(kwargs.get('workers', 1)),  # Number of threads that will be used for training.
                'optimize_interval': kwargs.get(
                    'optimize_interval', 10
                ),  # Optimize hyperparameters every optimize_interval iterations
                'topic_threshold': kwargs.get(
                    'topic_threshold', 0.0
                ),  # Threshold of the probability above which we consider a topic.
                'random_seed': kwargs.get(
                    'random_seed', 0
                ),  # Random seed to ensure consistent results, if 0 - use system clock.
            },
        }

    if algorithm.startswith('STTM-'):
        sttm = algorithm[5:]
        return {
            'engine': STTMTopicModel,
            'options': {
                'sstm_jar_path': './lib/STTM.jar',
                'model': sttm,
                'corpus': corpus,
                'id2word': id2word,
                'num_topics': kwargs.get('n_topics', 20),
                'iterations': kwargs.get('max_iter', 2000),
                'prefix': kwargs.get('prefix', TEMP_PATH),
                'name': '{}_model'.format(sttm)
                # 'vectors', 'alpha'=0.1, 'beta'=0.01, 'twords'=20,sstep=0
            },
        }

    raise ValueError('Unknown model!')
