import contextlib
import os
from typing import Literal, Mapping

from penelope.vendor.gensim_api import corpora as gensim_corpora
from penelope.vendor.gensim_api import models as gensim_models

from ..interface import EngineSpec

DEFAULT_WORK_FOLDER = './tmp/'

# pylint: disable=too-many-return-statements, inconsistent-return-statements


def get_int(d: dict, key: str, default: None) -> int:
    with contextlib.suppress(Exception):
        if key in d:
            return int(d[key]) if d[key] is not None else None
    return default


def get_float(d: dict, key: str, default: None) -> float:
    with contextlib.suppress(Exception):
        if key in d:
            return float(d[key]) if d[key] is not None else None
    return default


class MalletEngineSpec(EngineSpec):
    def __init__(self):
        super().__init__("gensim_mallet-lda", gensim_models.MalletTopicModel)

    def get_options(self, corpus: gensim_corpora.Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        work_folder: str = f"{engine_args.get('work_folder', DEFAULT_WORK_FOLDER).rstrip('/')}/mallet/"
        os.makedirs(work_folder, exist_ok=True)
        return dict(
            corpus=corpus,
            id2word=id2word,
            default_mallet_home='/usr/local/share/mallet-2.0.8/',
            prefix=work_folder,
            num_topics=get_int(engine_args, 'n_topics', 100),
            iterations=get_int(engine_args, 'max_iter', 3000),
            workers=get_int(engine_args, 'workers', 1),
            optimize_interval=get_int(engine_args, 'optimize_interval', 10),
            topic_threshold=get_float(engine_args, 'minimum_probability', 0.005),
            random_seed=get_int(engine_args, 'random_seed', 0),
            num_top_words=get_int(engine_args, 'num_top_words', 500),
            # 'alpha': int(kwargs.get('alpha', 20))
        )


class LdaEngineSpec(EngineSpec):
    def __init__(self):
        super().__init__("gensim_lda", gensim_models.LdaModel)

    def get_options(self, corpus: gensim_corpora.Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        return dict(
            # distributed=False, chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None, minimum_phi_value=0.01, per_word_topics=False, callbacks=None, dtype=<class 'numpy.float32'>)¶
            corpus=corpus,
            id2word=id2word,
            alpha=engine_args.get('alpha', 'symmetric'),
            chunksize=get_int(engine_args, 'chunk_size', 2000),
            iterations=get_int(engine_args, 'max_iter', 3000),
            num_topics=get_int(engine_args, 'n_topics', 100),
            passes=get_int(engine_args, 'passes', 1),
            per_word_topics=engine_args.get('per_word_topics', False),
            random_state=engine_args.get('random_state'),
            update_every=get_int(engine_args, 'update_every', 1),
            minimum_probability=engine_args.get('minimum_probability', 0.005),
            # decay= 0.1, # 0.5
            # offset=1.0,
            # dtype=np.float64
            # callbacks=[
            #    gensim_models.callbacks.PerplexityMetric(corpus=corpus, logger='visdom'),
            #    gensim_models.callbacks.ConvergenceMetric(distance='jaccard', num_words=100, logger='shell')
            # ]
        )


class LdaMulticoreEngineSpec(EngineSpec):

    """
    See Gensim documentation: https://radimrehurek.com/gensim/gensim_models/ldamulticore.html#gensim.gensim_models.ldamulticore.LdaMulticore

        alpha               = 'symmetric'
        batch               = False
        chunksize           = 2000
        corpus              = None
        decay               = 0.5
        dtype               = np.float32
        eta                 = None
        eval_every          = 10
        gamma_threshold     = 0.001
        id2word             = None
        iterations          = 50
        minimum_phi_value   = 0.01
        minimum_probability = 0.01
        num_topics          = 100
        offset              = 1.0
        passes              = 1
        per_word_topics     = False
        random_state        = None
        workers             = None

    Parameters
    corpus ({iterable of list of (int, float), scipy.sparse.csc}, optional)
        Stream of document vectors or sparse matrix of shape (num_documents, num_terms).
        If not given, the model is left untrained (presumably because you want to call update() manually).

    num_topics (int, optional)
        The number of requested latent topics to be extracted from the training corpus.

    id2word ({dict of (int, str), gensim.corpora.dictionary.Dictionary})
        Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for debugging and topic printing.

    workers (int, optional)
        Number of workers processes to be used for parallelization.
        If None all available cores (as estimated by workers=cpu_count()-1 will be used.
        Note however that for hyper-threaded CPUs, this estimation returns a too high number
        set workers directly to the number of your real cores (not hyperthreads) minus one, for optimal performance.

    chunksize (int, optional)
        Number of documents to be used in each training chunk.

    passes (int, optional)
        Number of passes through the corpus during training.

    alpha ({float, numpy.ndarray of float, list of float, str}, optional)
        A-priori belief on document-topic distribution, this can be:
            - scalar for a symmetric prior over document-topic distribution,
            - 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.
        Alternatively default prior selecting strategies can be employed by supplying a string:
            - `symmetric`: (default) Uses a fixed symmetric prior of 1.0 / num_topics,
            - `asymmetric`: Uses a fixed normalized asymmetric prior of 1.0 / (topic_index + sqrt(num_topics)).

    eta ({float, numpy.ndarray of float, list of float, str}, optional)
        A-priori belief on topic-word distribution, this can be:
            - scalar for a symmetric prior over topic-word distribution,
            - 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,
            - matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.
        Alternatively default prior selecting strategies can be employed by supplying a string:
            - `symmetric`: (default) Uses a fixed symmetric prior of 1.0 / num_topics,
            - `auto`: Learns an asymmetric prior from the corpus.

    decay (float, optional)
        A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten when each new document is examined.
        Corresponds to kappa from `Online Learning for LDA` by Hoffman et al.

    offset (float, optional)
        Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
        Corresponds to tau_0 from `Online Learning for LDA` by Hoffman et al.

    eval_every (int, optional)
        Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.

    iterations (int, optional)
        Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.

    gamma_threshold (float, optional)
        Minimum change in the value of the gamma parameters to continue iterating.

    minimum_probability (float, optional)
        Topics with a probability lower than this threshold will be filtered out.

    random_state ({np.random.RandomState, int}, optional)
        Either a randomState object or a seed to generate one. Useful for reproducibility.
        Note that results can still vary due to non-determinism in OS scheduling of the worker processes.

    minimum_phi_value (float, optional)
        if per_word_topics is True, this represents a lower bound on the term probabilities.

    per_word_topics (bool)
        If True, the model also computes a list of topics, sorted in descending order of most
        likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count).

    dtype ({numpy.float16, numpy.float32, numpy.float64}, optional)
        Data-type to use during calculations inside model. All inputs are also converted.

    """

    def __init__(self):
        super().__init__("gensim_lda-multicore", gensim_models.LdaMulticore)

    def get_options(self, corpus: gensim_corpora.Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        # https://stackoverflow.com/questions/65014553/how-to-tune-the-parameters-for-gensim-ldamulticore-in-python
        opts: dict = dict(
            corpus=corpus,
            id2word=id2word,
            alpha=engine_args.get('alpha', 'symmetric'),
            chunksize=get_int(engine_args, 'chunk_size', 2000),
            iterations=get_int(engine_args, 'max_iter', 3000),
            minimum_probability=engine_args.get('minimum_probability', 0.005),
            num_topics=get_int(engine_args, 'n_topics', 100),
            passes=get_int(engine_args, 'passes', 1),
            per_word_topics=engine_args.get('per_word_topics', False),
            random_state=get_int(engine_args, 'random_seed', None),
            workers=get_int(engine_args, 'workers', None),
        )
        if 'dtype' in engine_args:
            opts.update(dtype=engine_args['dtype'])
        return opts


class LsiEngineSpec(EngineSpec):
    def __init__(self):
        super().__init__("gensim_lsi", gensim_models.LsiModel)

    def get_options(self, corpus: gensim_corpora.Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        return dict(
            corpus=corpus,
            id2word=id2word,
            num_topics=get_int(engine_args, 'n_topics', 100),
            power_iters=2,
            onepass=True,
        )


class STTMEngineSpec(EngineSpec):
    def __init__(self, sub_key: Literal['lda', 'btm', 'ptm', 'satm', 'dmm', 'watm']):
        super().__init__(f"gensim_sttm-{sub_key}", gensim_models.STTMTopicModel)

    @property
    def sttm_type(self) -> str:
        return self.algorithm[5:]

    def get_options(self, corpus: gensim_corpora.Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        work_folder: str = f"{engine_args.get('work_folder', DEFAULT_WORK_FOLDER).rstrip('/')}/sttm/"
        return dict(
            sstm_jar_path='./lib/STTM.jar',
            model=self.sttm_type,
            corpus=corpus,
            id2word=id2word,
            num_topics=get_int(engine_args, 'n_topics', 100),
            iterations=get_int(engine_args, 'max_iter', 3000),
            prefix=work_folder,
            name=f'{self.sttm_type}_model',
            # 'vectors', 'alpha'=0.1, 'beta'=0.01, 'twords'=20,sstep=0
        )


# class HDPEngineSpec(EngineSpec):
#     def __init__(self):
#         super().__init__("gensim_hdp", gensim_models.HdpModel)

#     def get_options(self, corpus: gensim_corpora.Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
#         return {
#             'corpus': corpus,
#             'T': engine_args.get('n_topics', 0),
#             'id2word': id2word,
#             # 'iterations': kwargs.get('max_iter', 0),
#             # 'passes': kwargs.get('passes', 20),
#             # 'alpha': 'symmetric'
#         }


class DTMEngineSpec(EngineSpec):
    def __init__(self):
        super().__init__("gensim_dtm", gensim_models.LdaSeqModel)

    def get_options(self, corpus: gensim_corpora.Sparse2Corpus, id2word: dict, engine_args: dict) -> dict:
        return {
            'corpus': corpus,
            'num_topics': engine_args.get('n_topics', 0),
            'id2word': id2word,
            # 'time_slice': document_index.count_documents_in_index_by_pivot(documents, year_column),
            # 'initialize': 'gensim/own/ldamodel',
            # 'lda_model': model # if initialize='gensim'
            # 'lda_inference_max_iter': kwargs.get('max_iter', 0),
            # 'passes': kwargs.get('passes', 20),
            # 'alpha': 'symmetric'
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
    # 'gensim_hdp': HDPEngineSpec(),
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
