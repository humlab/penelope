import contextlib
import gzip
import importlib
import json
import os
import pickle
import time
from os.path import join as jj
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional

import numpy as np
import pandas as pd
import scipy
from penelope.utility import read_json, write_json

from ..document_index import DocumentIndex
from .interface import IVectorizedCorpus, IVectorizedCorpusProtocol


def create_corpus_instance(
    bag_term_matrix: scipy.sparse.csr_matrix,
    token2id: Dict[str, int],
    document_index: DocumentIndex,
    overridden_term_frequency: Dict[str, int] = None,
) -> "IVectorizedCorpus":
    """Creates a corpus instance using importlib to avoid cyclic references"""
    module = importlib.import_module(name="penelope.corpus.dtm.corpus")
    cls = getattr(module, "VectorizedCorpus")
    return cls(
        bag_term_matrix=bag_term_matrix,
        token2id=token2id,
        document_index=document_index,
        overridden_term_frequency=overridden_term_frequency,
    )


def matrix_filename(tag: str, folder: str, extension: str = '') -> str:
    """Returns BoW matrix basename for given tag and folder"""
    return jj(folder, f"{tag}_vector_data{extension}")


def load_metadata(*, tag: str, folder: str) -> dict:

    pickle_filename: str = jj(folder, f"{tag}_vectorizer_data.pickle")
    if os.path.isfile(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            data = pickle.load(f)
        return data

    if os.path.isfile(jj(folder, f"{tag}_document_index.csv.gz")):

        document_index: pd.DataFrame = pd.read_csv(
            jj(folder, f"{tag}_document_index.csv.gz"), sep=';', compression="gzip", index_col=0
        )

        with gzip.open(jj(folder, f"{tag}_token2id.json.gz"), 'r') as fp:
            token2id: dict = json.loads(fp.read().decode('utf-8'))

        term_frequency = (
            np.load(jj(folder, f"{tag}_overridden_term_frequency.npy"), allow_pickle=True)
            if os.path.isfile(jj(folder, f"{tag}_overridden_term_frequency.npy"))
            else None
        )

        return {
            'token2id': token2id,
            'document_index': document_index,
            'overridden_term_frequency': term_frequency,
        }

    raise ValueError("No metadata in folder")


def store_metadata(*, tag: str, folder: str, mode: Literal['bundle', 'files'] = 'bundle', **data) -> None:

    if mode.startswith('bundle'):

        pickle_filename: str = jj(folder, f"{tag}_vectorizer_data.pickle")
        with open(pickle_filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return

    if mode.startswith('files'):

        data.get('document_index').to_csv(jj(folder, f"{tag}_document_index.csv.gz"), sep=';', compression="gzip")

        with gzip.open(jj(folder, f"{tag}_token2id.json.gz"), 'w') as fp:  # 4. fewer bytes (i.e. gzip)
            fp.write(json.dumps(data.get('token2id')).encode('utf-8'))

        term_frequency: np.ndarray = data.get('overridden_term_frequency')
        if term_frequency is not None:
            np.save(jj(folder, f"{tag}_overridden_term_frequency.npy"), term_frequency, allow_pickle=True)

        return

    raise ValueError(f"Invalid mode {mode}")


class StoreMixIn:
    def dump(
        self: IVectorizedCorpusProtocol, *, tag: str, folder: str, compressed: bool = True, mode: str = 'bundle'
    ) -> IVectorizedCorpus:
        """Store corpus on disk.

        The file is stored as two files: one that contains the BoW matrix (.npy or .npz)
        and a pickled/gzipped file that contains dictionary, word counts and the document index

        The files are stored in files with names prefixed with the specified `tag`:

            {tag}_vectorizer_data.pickle         Bundle with `token2id`, `document_index` and `overridden_term_frequency`
            {tag}_document_index.csv.gz          Document index as compressed CSV (if mode is `files`)
            {tag}_token2id.json.gz               Vocabluary as compressed JSON (if mode is `files`)
            {tag}_term_frequency.npy             Term frequency to use, overrides TF sums in DTM (if mode is `files`)
            {tag}_vector_data.[npz|npy]          The document-term matrix (numpy or sparse format)


        Parameters
        ----------
        tag : str, optional
            String to be prepended to file name, set to timestamp if None
        folder : str, optional
            Target folder, by default './output'
        compressed : bool, optional
            Specifies if matrix is stored as .npz or .npy, by default .npz
        mode : str, optional, values 'bundle' or 'files'
            Specifies if metadata should be bundled in a pickle file or stored as individual compressed files.

        """
        tag = tag or time.strftime("%Y%m%d_%H%M%S")

        store_metadata(tag=tag, folder=folder, mode=mode, **self.metadata)

        if compressed:
            assert scipy.sparse.issparse(self.bag_term_matrix)
            scipy.sparse.save_npz(jj(folder, f"{tag}_vector_data"), self.bag_term_matrix, compressed=True)
        else:
            np.save(jj(folder, f"{tag}_vector_data.npy"), self.bag_term_matrix, allow_pickle=True)

        return self

    @property
    def metadata(self) -> dict:
        return {
            'token2id': self.token2id,
            'overridden_term_frequency': self.overridden_term_frequency,
            'document_index': self.document_index,
        }

    @staticmethod
    def dump_exists(*, tag: str, folder: str) -> bool:
        """Checks if corpus with tag `tag` exists in folder `folder`

        Parameters
        ----------
        tag : str
            Corpus prefix tag
        folder : str, optional
            Corpus folder to look in
        """
        return any(
            os.path.isfile(jj(folder, f"{tag}_{suffix}"))
            for suffix in [
                'vector_data.npz',
                'vector_data.npy',
                'vectorizer_data.pickle',
                'document_index.csv.gz',
            ]
        )

    @staticmethod
    def remove(*, tag: str, folder: str):
        with contextlib.suppress(Exception):
            Path(jj(folder, f'{tag}_vector_data.npz')).unlink(missing_ok=True)
            Path(jj(folder, f'{tag}_vector_data.npy')).unlink(missing_ok=True)
            Path(jj(folder, f"{tag}_vectorizer_data.json")).unlink(missing_ok=True)
            Path(jj(folder, f"{tag}_vectorizer_data.pickle")).unlink(missing_ok=True)
            Path(jj(folder, f"{tag}_document_index.csv.gz")).unlink(missing_ok=True)
            Path(jj(folder, f"{tag}_token2id.json.gz")).unlink(missing_ok=True)
            Path(jj(folder, f"{tag}_overridden_term_frequency.npy")).unlink(missing_ok=True)

    @staticmethod
    def load(*, tag: str, folder: str) -> IVectorizedCorpus:
        """Loads corpus with tag `tag` in folder `folder`

        Raises `FileNotFoundError` if any of the two files containing metadata and matrix doesn't exist.

        Two files are loaded based on specified `tag`:

            {tag}_vectorizer_data.pickle         Contains metadata `token2id`, `document_index` and `overridden_term_frequency`
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

        data: dict = load_metadata(tag=tag, folder=folder)

        token2id: Mapping = data.get("token2id")

        """Load TF override, convert if in older (dict) format"""
        overridden_term_frequency: np.ndarray = (
            data.get("term_frequency", None)
            or data.get("overridden_term_frequency", None)
            or data.get("term_frequency_mapping", None)
            or data.get("token_counter", None)
        )
        if isinstance(overridden_term_frequency, dict):
            fg = {v: k for k, v in token2id.items()}.get
            overridden_term_frequency = np.array([overridden_term_frequency[fg(i)] for i in range(0, len(token2id))])

        """Document-term-matrix"""
        if os.path.isfile(jj(folder, f"{tag}_vector_data.npz")):
            bag_term_matrix = scipy.sparse.load_npz(jj(folder, f"{tag}_vector_data.npz"))
        else:
            bag_term_matrix = np.load(jj(folder, f"{tag}_vector_data.npy"), allow_pickle=True).item()

        return create_corpus_instance(
            bag_term_matrix,
            token2id=token2id,
            document_index=data.get("document_index"),
            overridden_term_frequency=overridden_term_frequency,
        )

    @staticmethod
    def dump_options(*, tag: str, folder: str, options: Dict):
        json_filename = jj(folder, f"{tag}_vectorizer_data.json")
        write_json(json_filename, options, default=lambda _: "<not serializable>")

    @staticmethod
    def load_options(*, tag: str, folder: str) -> Dict:
        """Loads vectrize options if they exists"""
        json_filename = jj(folder, f"{tag}_vectorizer_data.json")
        if os.path.isfile(json_filename):
            return read_json(json_filename)
        return dict()


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
        v_corpus = v_corpus.slice_by_tf(n_count)

    if n_top is not None:
        v_corpus = v_corpus.slice_by_n_top(n_top)

    if axis is not None and v_corpus.data.shape[1] > 0:
        v_corpus = v_corpus.normalize(axis=axis, keep_magnitude=keep_magnitude)

    return v_corpus
