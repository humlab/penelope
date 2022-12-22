import os
import re
import xml.etree.ElementTree as ET
from typing import Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
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

    # def ftopicwordweights(self) -> str:
    #     return self.prefix + 'topicwordweights.txt'

    def diagnostics_filename(self) -> str:
        return self.prefix + 'diagnostics.xml'

    def train(self, corpus: Iterable[Iterable[Tuple[int, int]]], **kwargs):
        """Train Mallet LDA.
        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format
        """
        use_existing_corpus: bool = kwargs.get('use_existing_corpus', False)

        if os.path.isfile(self.mallet_corpus_filename()) and use_existing_corpus:
            logger.warning("using EXISTING corpus.mallet!")
        else:
            self.convert_input(corpus, infer=False)

        cmd: str = (
            f"{self.mallet_path} train-topics "
            f"--input {self.mallet_corpus_filename()} "
            f"--num-topics {self.num_topics} "
            f"--alpha {self.alpha} "
            f"--optimize-interval {self.optimize_interval} "
            f"--num-threads {self.workers} "
            f"--output-state {self.mallet_state_filename()} "
            f"--output-doc-topics {self.document_topics_filename()} "
            f"--output-topic-keys {self.topic_keys_filename()} "
            f"--num-top-words {self.num_top_words} "
            f"--diagnostics-file {self.diagnostics_filename()} "
            f"--num-iterations {self.iterations} "
            f"--inferencer-filename {self.inferencer_filename()} "
            f"--doc-topics-threshold {self.topic_threshold} "
            f"--random-seed {str(self.random_seed)} "
        )

        # f"--topic-word-weights-file {self.ftopicwordweights()} "

        logger.info(f"training MALLET LDA with {cmd}")
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

    def load_topic_diagnostics(self) -> pd.DataFrame:
        """Loads MALLET topic diagnostics item into dataframe
        See: https://mallet.cs.umass.edu/diagnostics.php
        """
        try:

            topics: pd.DataFrame = (
                pd.read_xml(self.diagnostics_filename(), xpath=".//topic")
                .rename(
                    columns={
                        'id': 'topic_id',
                        'tokens': 'n_tokens',
                    }
                )
                .set_index('topic_id')
            )

            return topics

        except Exception as ex:
            logger.error(f"load_topic_diagnostics: {ex}")
            return None

    def load_topic_token_diagnostics(self) -> pd.DataFrame:
        return MalletTopicModel.load_topic_token_diagnostics2(self.diagnostics_filename())

    @staticmethod
    def load_topic_token_diagnostics2(source: str) -> pd.DataFrame:
        """Loads MALLET word diagnostics item into dataframe
        See: https://mallet.cs.umass.edu/diagnostics.php
        """
        try:

            # words: pd.DataFrame = pd.read_xml(self.diagnostics_filename(), xpath=".//word")
            dtypes: dict = {
                'rank': np.int32,
                'count': np.int64,
                'prob': np.float64,
                'cumulative': np.float64,
                'docs': int,
                'word-length': np.float64,
                'coherence': np.float64,
                'uniform_dist': np.float64,
                'corpus_dist': np.float64,
                'token-doc-diff': np.float64,
                'exclusivity': np.float64,
                'topic_id': np.int16,
                'token': str,
            }

            words: pd.DataFrame = pd.DataFrame(data=[x for x in MalletTopicModel.parse_diagnostics_words(source)])

            for k, t in dtypes.items():
                if k in words.columns:
                    words[k] = words[k].astype(t)

            return words

        except Exception as ex:
            logger.error(f"load_topic_token_diagnostics: {ex}")
            return None

    @staticmethod
    def parse_diagnostics_words(source: str) -> Iterable:

        context = ET.iterparse(source, events=("start", "end"))

        topic_id: int = None
        item: dict = {}

        for event, elem in context:

            tag = elem.tag.rpartition('}')[2]

            if event == 'start' and tag == 'topic':
                topic_id = int(elem.attrib.get('id'))

            if tag == 'word':

                if event == 'end':
                    item = dict(elem.attrib)
                    item['topic_id'] = int(topic_id)
                    item['token'] = elem.text
                    yield item

            if event == 'end':
                elem.clear()
