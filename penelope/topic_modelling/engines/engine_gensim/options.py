import os
from typing import Literal, Mapping

import gensim.models as models
from gensim.matutils import Sparse2Corpus

from ..interface import EngineSpec
from .wrappers import MalletTopicModel, STTMTopicModel

DEFAULT_WORK_FOLDER = './tmp/'

# pylint: disable=too-many-return-statements, inconsistent-return-statements


class MalletEngineSpec(EngineSpec):
    def __init__(self):
        super().__init__("gensim_mallet-lda", MalletTopicModel)

    def get_options(self, corpus: Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        work_folder: str = f"{engine_args.get('work_folder', DEFAULT_WORK_FOLDER).rstrip('/')}/mallet/"
        os.makedirs(work_folder, exist_ok=True)
        return {
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
        }


class LdaEngineSpec(EngineSpec):
    def __init__(self):
        super().__init__("gensim_lda", models.LdaModel)

    def get_options(self, corpus: Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        return {
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
        }


class LdaMulticoreEngineSpec(EngineSpec):
    def __init__(self):
        super().__init__("gensim_lda-multicore", models.LdaMulticore)

    def get_options(self, corpus: Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        return {
            # workers=None, chunksize=2000, passes=1, batch=False, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, random_state=None, minimum_probability=0.01, minimum_phi_value=0.01, per_word_topics=False, dtype=<class 'numpy.float32'>)v
            'corpus': corpus,  # Stream of document vectors or sparse matrix of shape (num_terms, num_documents).
            'num_topics': int(engine_args.get('n_topics', 20)),
            'id2word': id2word,  # id2word ({dict of (int, str), gensim.corpora.dictionary.Dictionary})
            'iterations': engine_args.get(
                'max_iter', 3000
            ),  # Maximum number of iterations through the corpus when inferring the topic distribution of a corpus
            'passes': int(engine_args.get('passes', 1)),  # Number of passes through the corpus during training.
            'workers': engine_args.get(
                'workers', 3
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
        }


class LsiEngineSpec(EngineSpec):
    def __init__(self):
        super().__init__("gensim_lsi", models.LsiModel)

    def get_options(self, corpus: Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        return dict(
            corpus=corpus,
            num_topics=engine_args.get('n_topics', 0),
            id2word=id2word,
            power_iters=2,
            onepass=True,
        )


class STTMEngineSpec(EngineSpec):
    def __init__(self, sub_key: Literal['lda', 'btm', 'ptm', 'satm', 'dmm', 'watm']):
        super().__init__(f"gensim_sttm-{sub_key}", STTMTopicModel)

    @property
    def sttm_type(self) -> str:
        return self.algorithm[5:]

    def get_options(self, corpus: Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        work_folder: str = f"{engine_args.get('work_folder', DEFAULT_WORK_FOLDER).rstrip('/')}/sttm/"
        return {
            'sstm_jar_path': './lib/STTM.jar',
            'model': self.sttm_type,
            'corpus': corpus,
            'id2word': id2word,
            'num_topics': engine_args.get('n_topics', 20),
            'iterations': engine_args.get('max_iter', 2000),
            'prefix': work_folder,
            'name': '{}_model'.format(self.sttm_type)
            # 'vectors', 'alpha'=0.1, 'beta'=0.01, 'twords'=20,sstep=0
        }


class HDPEngineSpec(EngineSpec):
    def __init__(self):
        super().__init__("gensim_hdp", models.HdpModel)

    def get_options(self, corpus: Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        return {
            'corpus': corpus,
            'T': engine_args.get('n_topics', 0),
            'id2word': id2word,
            # 'iterations': kwargs.get('max_iter', 0),
            # 'passes': kwargs.get('passes', 20),
            # 'alpha': 'auto'
        }


class DTMEngineSpec(EngineSpec):
    def __init__(self):
        super().__init__("gensim_dtm", models.LdaSeqModel)

    def get_options(self, corpus: Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        return {
            'corpus': corpus,
            'num_topics': engine_args.get('n_topics', 0),
            'id2word': id2word,
            # 'time_slice': document_index.count_documents_in_index_by_pivot(documents, year_column),
            # 'initialize': 'gensim/own/ldamodel',
            # 'lda_model': model # if initialize='gensim'
            # 'lda_inference_max_iter': kwargs.get('max_iter', 0),
            # 'passes': kwargs.get('passes', 20),
            # 'alpha': 'auto'
        }


EngineKey = Literal[
    'gensim_mallet-lda',
    'gensim_lda-multicore',
    'gensim_lda',
    'gensim_lsi',
    'gensim_hdp',
    'gensim_dtm',
    'gensim_sttm-lda',
    'gensim_sttm-btm',
    'gensim_sttm-ptm',
    'gensim_sttm-satm',
    'gensim_sttm-dmm',
    'gensim_sttm-watm',
]

SUPPORTED_ENGINES: Mapping[EngineKey, EngineSpec] = {
    'gensim_mallet-lda': MalletEngineSpec(),
    'gensim_lda-multicore': LdaMulticoreEngineSpec(),  # 'LDA-MULTICORE', 'LDA_MULTICORE', 'MULTICORE'
    'gensim_lda': LdaEngineSpec(),
    'gensim_lsi': LsiEngineSpec(),
    'gensim_hdp': HDPEngineSpec(),
    'gensim_dtm': DTMEngineSpec(),
    'gensim_sttm-lda': STTMEngineSpec(sub_key='lda'),
    'gensim_sttm-btm': STTMEngineSpec(sub_key='btm'),
    'gensim_sttm-ptm': STTMEngineSpec(sub_key='ptm'),
    'gensim_sttm-satm': STTMEngineSpec(sub_key='satm'),
    'gensim_sttm-dmm': STTMEngineSpec(sub_key='dmm'),
    'gensim_sttm-watm': STTMEngineSpec(sub_key='watm'),
}


def get_engine_specification(*, engine_key: EngineKey) -> EngineSpec:
    return SUPPORTED_ENGINES.get(engine_key) or SUPPORTED_ENGINES.get(f'gensim_{engine_key}'.lower())
