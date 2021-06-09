import importlib
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import scipy
from penelope.utility import read_json, write_json

from ..document_index import DocumentIndex
from .interface import IVectorizedCorpus, IVectorizedCorpusProtocol


def create_corpus_instance(
    bag_term_matrix: scipy.sparse.csr_matrix,
    token2id: Dict[str, int],
    document_index: DocumentIndex,
    term_frequency_mapping: Dict[str, int] = None,
) -> "IVectorizedCorpus":
    """Creates a corpus instance using importlib to avoid cyclic references"""
    module = importlib.import_module(name="penelope.corpus.dtm.vectorized_corpus")
    cls = getattr(module, "VectorizedCorpus")
    return cls(
        bag_term_matrix=bag_term_matrix,
        token2id=token2id,
        document_index=document_index,
        term_frequency_mapping=term_frequency_mapping,
    )


class StoreMixIn:
    def dump(self: IVectorizedCorpusProtocol, *, tag: str, folder: str, compressed: bool = True) -> IVectorizedCorpus:
        """Store corpus on disk.

        The file is stored as two files: one that contains the BoW matrix (.npy or .npz)
        and a pickled file that contains dictionary, word counts and the document index

        The two files are stored in files with names based on the specified `tag`:

            {tag}_vectorizer_data.pickle         Metadata `token2id`, `document_index` and `term_frequency_mapping`
            {tag}_vector_data.[npz|npy]          The document-term matrix (numpy or sparse format)


        Parameters
        ----------
        tag : str, optional
            String to be prepended to file name, set to timestamp if None
        folder : str, optional
            Target folder, by default './output'
        compressed : bool, optional
            Specifies if matrix is stored as .npz or .npy, by default .npz

        """
        tag = tag or time.strftime("%Y%m%d_%H%M%S")

        data = {
            'token2id': self.token2id,
            'term_frequency_mapping': self.term_frequency_mapping,
            'document_index': self.document_index,
        }
        data_filename = StoreMixIn._data_filename(tag, folder)

        with open(data_filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        matrix_filename = StoreMixIn._matrix_filename(tag, folder)

        if compressed:
            assert scipy.sparse.issparse(self.bag_term_matrix)
            scipy.sparse.save_npz(matrix_filename, self.bag_term_matrix, compressed=True)
        else:
            np.save(matrix_filename + '.npy', self.bag_term_matrix, allow_pickle=True)

        return self

    @staticmethod
    def dump_exists(*, tag: str, folder: str) -> bool:
        """Checks if corpus with tag `tag` exists in folder `folder`

        Parameters
        ----------
        tag : str
            Corpus prefix tag
        folder : str, optional
            Corpus folder to look in, by default './output'
        """
        return os.path.isfile(StoreMixIn._data_filename(tag, folder))

    @staticmethod
    def remove(*, tag: str, folder: str):
        Path(os.path.join(folder, f'{tag}_vector_data.npz')).unlink(missing_ok=True)
        Path(os.path.join(folder, f'{tag}_vector_data.npy')).unlink(missing_ok=True)
        Path(os.path.join(folder, f'{tag}_vectorizer_data.pickle')).unlink(missing_ok=True)
        Path(os.path.join(folder, f'{tag}_vectorizer_data.json')).unlink(missing_ok=True)

    @staticmethod
    def load(*, tag: str, folder: str) -> IVectorizedCorpus:
        """Loads corpus with tag `tag` in folder `folder`

        Raises `FileNotFoundError` if any of the two files containing metadata and matrix doesn't exist.

        Two files are loaded based on specified `tag`:

            {tag}_vectorizer_data.pickle         Contains metadata `token2id`, `document_index` and `term_frequency_mapping`
            {tag}_vector_data.[npz|npy]          Contains the document-term matrix (numpy or sparse format)


        Parameters
        ----------
        tag : str
            Corpus prefix tag
        folder : str, optional
            Corpus folder to look in, by default './output'

        Returns
        -------
        VectorizedCorpus
            Loaded corpus
        """
        data_filename = StoreMixIn._data_filename(tag, folder)
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)

        token2id: Mapping = data["token2id"]
        document_index: DocumentIndex = data.get("document_index")
        term_frequency_mapping: dict = data.get("term_frequency_mapping", data.get("token_counter", None))
        matrix_basename = StoreMixIn._matrix_filename(tag, folder)

        if os.path.isfile(matrix_basename + '.npz'):
            bag_term_matrix = scipy.sparse.load_npz(matrix_basename + '.npz')
        else:
            bag_term_matrix = np.load(matrix_basename + '.npy', allow_pickle=True).item()

        return create_corpus_instance(
            bag_term_matrix,
            token2id=token2id,
            document_index=document_index,
            term_frequency_mapping=term_frequency_mapping,
        )

    @staticmethod
    def dump_options(*, tag: str, folder: str, options: Dict):
        json_filename = os.path.join(folder, f"{tag}_vectorizer_data.json")
        write_json(json_filename, options, default=lambda _: "<not serializable>")

    @staticmethod
    def load_options(*, tag: str, folder: str) -> Dict:
        """Loads vectrize options if they exists"""
        json_filename = os.path.join(folder, f"{tag}_vectorizer_data.json")
        if os.path.isfile(json_filename):
            return read_json(json_filename)
        return dict()

    @staticmethod
    def _data_filename(tag: str, folder: str) -> str:
        """Returns pickled basename for given tag and folder"""
        return os.path.join(folder, f"{tag}_vectorizer_data.pickle")

    @staticmethod
    def _matrix_filename(tag: str, folder: str) -> str:
        """Returns BoW matrix basename for given tag and folder"""
        return os.path.join(folder, f"{tag}_vector_data")


def load_corpus(
    *,
    tag: str,
    folder: str,
    n_count: int = 10000,
    n_top: int = 100000,
    axis: Optional[int] = 1,
    keep_magnitude: bool = True,
    group_by_year: bool = True,
) -> IVectorizedCorpus:
    """Loads a previously saved vectorized corpus from disk. Easaly the best loader ever.

    Parameters
    ----------
    tag : str
        Corpus filename prefix
    folder : str
        Source folder where corpus reside
    n_count : int, optional
        Words having a (global) count below this limit are discarded, by default 10000
    n_top : int, optional
        Only the 'n_top' words sorted by word counts should be loaded, by default 100000
    axis : int, optional
        Axis used to normalize the data along. 1 normalizes each row (bag/document), 0 normalizes each column (word).
    keep_magnitude : bool, optional
        Scales result matrix so that sum equals input matrix sum, by default True

    Returns
    -------
    VectorizedCorpus
        The loaded corpus
    """
    v_corpus = StoreMixIn.load(tag=tag, folder=folder)

    if group_by_year:
        v_corpus = v_corpus.group_by_year()

    if n_count is not None:
        v_corpus = v_corpus.slice_by_n_count(n_count)

    if n_top is not None:
        v_corpus = v_corpus.slice_by_n_top(n_top)

    if axis is not None and v_corpus.data.shape[1] > 0:
        v_corpus = v_corpus.normalize(axis=axis, keep_magnitude=keep_magnitude)

    return v_corpus
