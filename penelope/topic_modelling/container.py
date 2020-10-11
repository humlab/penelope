import os
import pickle
import sys
import types
from typing import Any, List, Tuple

import pandas as pd

import penelope.utility as utility

from .utility import add_document_metadata

logger = utility.getLogger('corpus_text_analysis')


class ModelAgnosticDataContainer:
    """Container for a topic model as a generic set of pd.DataFrames. The content is common to all types model engines."""

    def __init__(
        self,
        documents: pd.DataFrame,                    # documents (training, shuould be predicted?)
        dictionary: Any,                            # dictionary
        topic_token_weights: pd.DataFrame,          # model data
        topic_token_overview: pd.DataFrame,         # model data
        document_topic_weights: pd.DataFrame,       # predicted data
    ):
        """A container for compiled data as generic pandas dataframes suitable for analysi and visualisation
        Parameters
        ----------
        documents : pd.DataFrame
            Corpus document index
        dictionary : Any
            Corpus dictionary
        topic_token_weights : pd.DataFrame
            Topic token weights
        topic_token_overview : pd.DataFrame
            Topic overview
        document_topic_weights : pd.DataFrame
            Document topic weights
        """
        self.dictionary = dictionary
        self.documents = documents
        self.topic_token_weights = topic_token_weights
        self.topic_token_overview = topic_token_overview
        self.document_topic_weights = document_topic_weights

        # Ensure that `year` column exists

        self.document_topic_weights = add_document_metadata(self.document_topic_weights, 'year', documents)

    @property
    def year_period(self) -> Tuple[int, int]:
        """Returns documents `year` interval (if exists)"""
        if 'year' not in self.document_topic_weights.columns:
            return (None, None)
        return (self.document_topic_weights.year.min(), self.document_topic_weights.year.max())

    @property
    def topic_ids(self) -> List[int]:
        """Returns unique topic ids """
        return list(self.document_topic_weights.topic_id.unique())

    def store(self, data_folder, model_name, pickled=False):
        """Stores aggregate in `data_folder` with filenames prefixed by `model_name`

        Parameters
        ----------
        data_folder : str
            target folder
        model_name : str
            Model name
        pickled : bool, optional
            if True then pickled (binary) format else  CSV, by default False
        """
        target_folder = os.path.join(data_folder, model_name)

        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)

        if pickled:

            filename = os.path.join(target_folder, "compiled_data.pickle")

            c_data = types.SimpleNamespace(
                documents=self.documents,
                dictionary=self.dictionary,
                topic_token_weights=self.topic_token_weights,
                topic_token_overview=self.topic_token_overview,
                document_topic_weights=self.document_topic_weights,
            )
            with open(filename, 'wb') as f:
                pickle.dump(c_data, f, pickle.HIGHEST_PROTOCOL)

        else:

            self.documents.to_csv(os.path.join(target_folder, 'documents.zip'), '\t')
            self.dictionary.to_csv(os.path.join(target_folder, 'dictionary.zip'), '\t')
            self.topic_token_weights.to_csv(os.path.join(target_folder, 'topic_token_weights.zip'), '\t')
            self.topic_token_overview.to_csv(os.path.join(target_folder, 'topic_token_overview.zip'), '\t')
            self.document_topic_weights.to_csv(os.path.join(target_folder, 'document_topic_weights.zip'), '\t')

    @staticmethod
    def load(folder: str, pickled: bool = False):
        """Loads previously stored aggregate"""
        data = None

        if pickled:

            filename = os.path.join(folder, "compiled_data.pickle")

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            data = ModelAgnosticDataContainer(
                data.documents,
                data.dictionary,
                data.topic_token_weights,
                data.topic_token_overview,
                data.document_topic_weights,
            )

        else:
            data = ModelAgnosticDataContainer(
                pd.read_csv(os.path.join(folder, 'documents.zip'), '\t', header=0, index_col=0, na_filter=False),
                pd.read_csv(os.path.join(folder, 'dictionary.zip'), '\t', header=0, index_col=0, na_filter=False),
                pd.read_csv(
                    os.path.join(folder, 'topic_token_weights.zip'), '\t', header=0, index_col=0, na_filter=False
                ),
                pd.read_csv(
                    os.path.join(folder, 'topic_token_overview.zip'), '\t', header=0, index_col=0, na_filter=False
                ),
                pd.read_csv(
                    os.path.join(folder, 'document_topic_weights.zip'), '\t', header=0, index_col=0, na_filter=False
                ),
            )

        return data

    def info(self):
        for o_name in [k for k in self.__dict__ if not k.startswith("__")]:
            o_data = getattr(self, o_name)
            o_size = sys.getsizeof(o_data)
            print('{:>20s}: {:.4f} Mb {}'.format(o_name, o_size / (1024 * 1024), type(o_data)))

    @property
    def id2term(self):
        return self.dictionary.token.to_dict()

    @property
    def term2id(self):
        return {v: k for k, v in self.id2term.items()}
