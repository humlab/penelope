import os
import re
from typing import Iterable, Mapping, Tuple

from loguru import logger

from penelope.utility import inspect_filter_args

from .ldamallet import LdaMallet

try:
    from gensim.utils import check_output
except ImportError:
    ...


class MalletTopicModel(LdaMallet):
    """Python wrapper for LDA using `MALLET <http://mallet.cs.umass.edu/>`_.
    This is a derived file of gensim.models.gensim_api.LdaMallet
    The following has been added:
       - Use of --topic-word-weights-file has been added
    """

    def __init__(
        self,
        corpus: Iterable[Iterable[Tuple[int, int]]],
        id2word: Mapping[int, str],
        default_mallet_home: str = None,
        **args,
    ):

        args: dict = inspect_filter_args(super().__init__, args)

        mallet_home: str = os.environ.get('MALLET_HOME', default_mallet_home)

        if not mallet_home:
            raise Exception("Environment variable MALLET_HOME not set. Aborting")

        mallet_path: str = os.path.join(mallet_home, 'bin', 'mallet') if mallet_home else None

        if os.environ.get('MALLET_HOME', '') != mallet_home:
            os.environ["MALLET_HOME"] = mallet_home

        super().__init__(mallet_path, corpus=corpus, id2word=id2word, **args)

    def ftopicwordweights(self) -> str:
        return self.prefix + 'topicwordweights.txt'

    def train(self, corpus: Iterable[Iterable[Tuple[int, int]]], **kwargs):
        """Train Mallet LDA.
        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format
        """
        use_existing_corpus: bool = kwargs.get('use_existing_corpus', False)

        if os.path.isfile(self.fcorpusmallet()) and use_existing_corpus:
            logger.warning("using EXISTING corpus.mallet!")
        else:
            self.convert_input(corpus, infer=False)

        cmd = (
            self.mallet_path + ' train-topics --input %s --num-topics %s  --alpha %s --optimize-interval %s '
            '--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s --topic-word-weights-file %s '
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s  --random-seed %s'
        )

        cmd = cmd % (
            self.fcorpusmallet(),
            self.num_topics,
            self.alpha,
            self.optimize_interval,
            self.workers,
            self.fstate(),
            self.fdoctopics(),
            self.ftopickeys(),
            self.ftopicwordweights(),
            self.iterations,
            self.finferencer(),
            self.topic_threshold,
            str(self.random_seed),
        )

        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()

        self.wordtopics = self.word_topics

    def xlog_perplexity(self, content: str) -> float:

        perplexity = None
        try:
            # content = open(filename).read()
            p = re.compile(r"<\d+> LL/token\: (-[\d\.]+)")
            matches = p.findall(content)
            if len(matches) > 0:
                perplexity = float(matches[-1])
        finally:
            return perplexity  # pylint: disable=lost-exception
