import logging
import os
import re

from gensim import models
from gensim.utils import check_output
from penelope.utility import inspect_filter_args

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class MalletTopicModel(models.wrappers.LdaMallet):
    """Python wrapper for LDA using `MALLET <http://mallet.cs.umass.edu/>`_.
    This is a derived file of gensim.models.wrappers.LdaMallet
    The following has been added:
        1. Use of --topic-word-weights-file has been added
    """

    def __init__(self, corpus, id2word, default_mallet_home=None, **args):

        args = inspect_filter_args(super().__init__, args)

        mallet_home = os.environ.get('MALLET_HOME', default_mallet_home)

        if not mallet_home:
            raise Exception("Environment variable MALLET_HOME not set. Aborting")

        mallet_path = os.path.join(mallet_home, 'bin', 'mallet') if mallet_home else None

        if os.environ.get('MALLET_HOME', '') != mallet_home:
            os.environ["MALLET_HOME"] = mallet_home

        super().__init__(mallet_path, corpus=corpus, id2word=id2word, **args)

    def ftopicwordweights(self):
        return self.prefix + 'topicwordweights.txt'

    def train(self, corpus):
        """Train Mallet LDA.
        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format
        """
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
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

    def xlog_perplexity(self, content):

        perplexity = None
        try:
            # content = open(filename).read()
            p = re.compile(r"<\d+> LL/token\: (-[\d\.]+)")
            matches = p.findall(content)
            if len(matches) > 0:
                perplexity = float(matches[-1])
        finally:
            return perplexity  # pylint: disable=lost-exception

    # def check_output(self, stdout=subprocess.PIPE, *popenargs, **kwargs):
    #     r"""Run OS command with the given arguments and return its output as a byte string.

    #     Backported from Python 2.7 with a few minor modifications. Widely used for :mod:`gensim.models.wrappers`.
    #     Behaves very similar to https://docs.python.org/2/library/subprocess.html#subprocess.check_output.

    #     Examples
    #     --------
    #     .. sourcecode:: pycon

    #         >>> from gensim.utils import check_output
    #         >>> check_output(args=['echo', '1'])
    #         '1\n'

    #     Raises
    #     ------
    #     KeyboardInterrupt
    #         If Ctrl+C pressed.

    #     """
    #     try:
    #         logger.debug("COMMAND: %s %s", popenargs, kwargs)
    #         process = subprocess.Popen(stdout=stdout, *popenargs, **kwargs)
    #         while True:

    #         output, unused_err = process.communicate()
    #         retcode = process.poll()
    #         if retcode:
    #             cmd = kwargs.get("args")
    #             if cmd is None:
    #                 cmd = popenargs[0]
    #             error = subprocess.CalledProcessError(retcode, cmd)
    #             error.output = output
    #             raise error
    #         return output
    #     except KeyboardInterrupt:
    #         process.terminate()
    #         raise

    # import io
    # import subprocess

    # proc = subprocess.Popen(["prog", "arg"], stdout=subprocess.PIPE)
    # for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):  # or another encoding
    #     # do something with line
