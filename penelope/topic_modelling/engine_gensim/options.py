import os
from typing import Literal

import gensim.models as models
from gensim.matutils import Sparse2Corpus

from .wrappers import MalletTopicModel, STTMTopicModel

DEFAULT_WORK_FOLDER = './tmp/'

# OBS OBS! https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
# DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)

# default_options = {
#     'LSI': {'engine': models.LsiModel, 'options': {'corpus': None, 'num_topics': 20, 'id2word': None}}
# }

SUPPORTED_ENGINES = {
    'gensim_mallet-lda': {
        'key': 'gensim_mallet-lda',
        'description': 'MALLET LDA',
        'engine': MalletTopicModel,
        'algorithm': 'MALLET-LDA',
    },
    'gensim_lda-multicore': {
        'key': 'gensim_lda',
        'description': 'gensim LDA',
        'engine': models.LdaMulticore,
        'algorithm': 'LDA-MULTICORE',
    },
    'gensim_lda': {
        'key': 'gensim_lda',
        'description': 'gensim LDA',
        'engine': models.LdaModel,
        'algorithm': 'LDA',
    },
    'gensim_lsi': {'key': 'gensim_lsi', 'description': 'gensim LSI', 'engine': models.LsiModel, 'algorithm': 'LSI'},
    'gensim_hdp': {'key': 'gensim_hdp', 'description': 'gensim HDP', 'engine': models.HdpModel, 'algorithm': 'HDP'},
    'gensim_dtm': {'key': 'gensim_dtm', 'description': 'gensim DTM', 'engine': models.LdaSeqModel, 'algorithm': 'DTM'},
    # 'sklearn_lda': {'key': 'sklearn_lda', 'description': 'scikit LDA', 'engine': None, 'algorithm': 'XXX'},
    # 'sklearn_nmf': {'key': 'sklearn_nmf', 'description': 'scikit NMF', 'engine': None, 'algorithm': 'XXX'},
    # 'sklearn_lsa': {'key': 'sklearn_lsa', 'description': 'scikit LSA', 'engine': None, 'algorithm': 'XXX'},
    # 'gensim_sttm-lda': {'key': 'gensim_sttm-lda', 'description': 'STTM   LDA', 'engine': STTMTopicModel, 'algorithm': 'STTM-LDA'},
    # 'gensim_sttm-btm': {'key': 'gensim_sttm-btm', 'description': 'STTM   BTM', 'engine': STTMTopicModel, 'algorithm': 'STTM-BTM'},
    # 'gensim_sttm-ptm': {'key': 'gensim_sttm-ptm', 'description': 'STTM   PTM', 'engine': STTMTopicModel, 'algorithm': 'STTM-PTM'},
    # 'gensim_sttm-satm': {'key': 'gensim_sttm-satm', 'description': 'STTM  SATM', 'engine': STTMTopicModel, 'algorithm': 'STTM-SATM'},
    # 'gensim_sttm-dmm': {'key': 'gensim_sttm-dmm', 'description': 'STTM   DMM', 'engine': STTMTopicModel, 'algorithm': 'STTM-DMM'},
    # 'gensim_sttm-watm': {'key': 'gensim_sttm-watm', 'description': 'STTM  WATM', 'engine': STTMTopicModel, 'algorithm': 'STTM-ATM'},
}

EngineKey = Literal[list(SUPPORTED_ENGINES.keys())]


# pylint: disable=too-many-return-statements, inconsistent-return-statements
def get_engine_options(
    *,
    algorithm: str,
    corpus: Sparse2Corpus,
    id2word: dict,
    engine_args: dict,
) -> dict:

    if algorithm == 'LSI':
        return {
            'engine': models.LsiModel,
            'options': {
                'corpus': corpus,
                'num_topics': engine_args.get('n_topics', 0),
                'id2word': id2word,
                'power_iters': 2,
                'onepass': True,
            },
        }

    if algorithm == 'LDA':
        return {
            'engine': models.LdaModel,
            'options': {
                # distributed=False, chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None, minimum_phi_value=0.01, per_word_topics=False, callbacks=None, dtype=<class 'numpy.float32'>)¶
                'corpus': corpus,
                'num_topics': int(engine_args.get('n_topics', 20)),
                'id2word': id2word,
                'iterations': engine_args.get('max_iter', 3000),
                'passes': int(engine_args.get('passes', 1)),
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
                #    models.callbacks.PerplexityMetric(corpus=corpus, logger='visdom'),
                #    models.callbacks.ConvergenceMetric(distance='jaccard', num_words=100, logger='shell')
                # ]
            },
        }

    if algorithm == 'LDA-MULTICORE':
        return {
            'engine': models.LdaMulticore,
            'options': {
                # workers=None, chunksize=2000, passes=1, batch=False, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, random_state=None, minimum_probability=0.01, minimum_phi_value=0.01, per_word_topics=False, dtype=<class 'numpy.float32'>)v
                'corpus': corpus,  # Sream of document vectors or sparse matrix of shape (num_terms, num_documents).
                'num_topics': int(engine_args.get('n_topics', 20)),
                'id2word': id2word,  # id2word ({dict of (int, str), gensim.corpora.dictionary.Dictionary})
                'iterations': engine_args.get(
                    'max_iter', 3000
                ),  # Maximum number of iterations through the corpus when inferring the topic distribution of a corpus
                'passes': int(engine_args.get('passes', 1)),  # Number of passes through the corpus during training.
                'workers': engine_args.get(
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
                #    models.callbacks.PerplexityMetric(corpus=corpus, logger='visdom'),
                #    models.callbacks.ConvergenceMetric(distance='jaccard', num_words=100, logger='shell')
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
            'engine': models.HdpModel,
            'options': {
                'corpus': corpus,
                'T': engine_args.get('n_topics', 0),
                'id2word': id2word,
                # 'iterations': kwargs.get('max_iter', 0),
                # 'passes': kwargs.get('passes', 20),
                # 'alpha': 'auto'
            },
        }

    if algorithm == 'DTM':
        # Note, mandatory: 'time_slice': document_index.count_documents_in_index_by_pivot(documents, year_column)
        return {
            'engine': models.LdaSeqModel,
            'options': {
                'corpus': corpus,
                'num_topics': engine_args.get('n_topics', 0),
                'id2word': id2word,
                # 'time_slice': document_index.count_documents_in_index_by_pivot(documents, year_column),
                # 'initialize': 'gensim/own/ldamodel',
                # 'lda_model': model # if initialize='gensim'
                # 'lda_inference_max_iter': kwargs.get('max_iter', 0),
                # 'passes': kwargs.get('passes', 20),
                # 'alpha': 'auto'
            },
        }

    if algorithm == 'MALLET-LDA':
        work_folder: str = f"{engine_args.get('work_folder', DEFAULT_WORK_FOLDER).rstrip('/')}/mallet/"
        os.makedirs(work_folder, exist_ok=True)
        return {
            'engine': MalletTopicModel,
            'options': {
                'corpus': corpus,  # Collection of texts in BoW format.
                'id2word': id2word,  # Dictionary
                'default_mallet_home': '/usr/local/share/mallet-2.0.8/',  # MALLET_HOME
                'num_topics': engine_args.get('n_topics', 100),  # Number of topics.
                'iterations': engine_args.get('max_iter', 3000),  # Number of training iterations.
                # 'alpha': int(kwargs.get('alpha', 20))                     # Alpha parameter of LDA.
                'prefix': work_folder,
                'workers': int(engine_args.get('workers', 1)),  # Number of threads that will be used for training.
                'optimize_interval': engine_args.get(
                    'optimize_interval', 10
                ),  # Optimize hyperparameters every optimize_interval iterations
                'topic_threshold': engine_args.get(
                    'topic_threshold', 0.0
                ),  # Threshold of the probability above which we consider a topic.
                'random_seed': engine_args.get(
                    'random_seed', 0
                ),  # Random seed to ensure consistent results, if 0 use system clock.
            },
        }

    if algorithm.startswith('STTM-'):
        work_folder: str = f"{engine_args.get('work_folder', DEFAULT_WORK_FOLDER).rstrip('/')}/sttm/"
        sttm = algorithm[5:]
        return {
            'engine': STTMTopicModel,
            'options': {
                'sstm_jar_path': './lib/STTM.jar',
                'model': sttm,
                'corpus': corpus,
                'id2word': id2word,
                'num_topics': engine_args.get('n_topics', 20),
                'iterations': engine_args.get('max_iter', 2000),
                'prefix': work_folder,
                'name': '{}_model'.format(sttm)
                # 'vectors', 'alpha'=0.1, 'beta'=0.01, 'twords'=20,sstep=0
            },
        }

    raise ValueError('Unknown model!')
