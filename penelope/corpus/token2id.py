import pathlib
import zipfile
from collections import defaultdict
from collections.abc import MutableMapping
from fnmatch import fnmatch
from typing import Iterator, List, Optional, Union

import pandas as pd
from loguru import logger
from penelope.utility import replace_extension, strip_paths


class Token2Id(MutableMapping):
    """A token-to-id mapping (dictionary)"""

    def __init__(self, data: Optional[Union[dict, defaultdict]] = None, lowercase: bool = False):
        if isinstance(data, defaultdict):
            self.data = data
        elif isinstance(data, dict):
            self.data = defaultdict(int, data)
        else:
            self.data = data or defaultdict()
        self.lowercase: bool = lowercase
        self.data.default_factory = self.data.__len__
        self._id2token: dict = None

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key: str, value):
        if self._id2token:
            self._id2token = None
        if self.lowercase:
            key = key.lower()
        self.data[key] = value

    def __delitem__(self, key):
        if self.lowercase:
            key = key.lower()
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def ingest(self, tokens: Iterator[str]) -> "Token2Id":
        self._id2token = None
        if self.lowercase:
            for token in tokens:
                _ = self.data[token.lower()]
        else:
            for token in tokens:
                _ = self.data[token]
        return self

    def is_open(self) -> bool:
        return self.data.default_factory is not None

    def close(self) -> "Token2Id":
        self.data.default_factory = None

    def open(self) -> "Token2Id":
        self.data.default_factory = self.__len__
        self._id2token = None
        return self

    @property
    def id2token(self) -> dict:
        # FIXME: Always create new reversed mapping if vocabulay is open
        if self._id2token is None or len(self) != len(self._id2token):  # or self.is_open():
            self._id2token = {v: k for k, v in self.data.items()}
        return self._id2token

    def to_dataframe(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame({'token': self.data.keys(), 'token_id': self.data.values()}).set_index('token')
        return df

    def store(self, filename: str):
        """Store dictionary as CSV"""
        # pandas_to_csv_zip(filename, dfs=(self.to_dataframe(), strip_paths(filename)), sep='\t', header=True)
        with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED) as fp:
            data_str = self.to_dataframe().to_csv(sep='\t', header=True)
            fp.writestr(replace_extension(strip_paths(filename), ".csv"), data=data_str)

    @staticmethod
    def load(filename: str) -> "Token2Id":
        """Store dictionary as CSV"""
        if not pathlib.Path(filename).exists():
            logger.info("bundle has no vocabulary")
            return None
        df: pd.DataFrame = pd.read_csv(filename, sep='\t', index_col=0, na_filter=False)
        data: dict = df['token_id'].to_dict()
        return Token2Id(data=data)

    def to_ids(self, tokens: List[str]) -> List[int]:
        return [self.data[w] for w in tokens]

    def find(self, what: Union[List[str], str]):

        if not what:
            return []

        if isinstance(what, (int, str)):
            what = [what]

        wildcards = [w for w in what if '*' in w]
        tokens = [w for w in what if w not in wildcards]

        matches = []

        if tokens:
            matches.extend([w for w in tokens if w in self.data])

        if wildcards:
            matches.extend([w for w in self.data.keys() if any(fnmatch(w, x) for x in wildcards)])

        return [self[w] for w in set(matches)]

    # def to_bow(self, documents: Iterator[Iterator[str]]):

    #     was_closed = not self.is_open()

    #     self.open()

    #     token2id = self.data
    #     counter = defaultdict(int)

    #     for w in document:
    #         counter[token2id[w]] += 1

    #     if was_closed:
    #         self.close()

    #     # result = sorted(result.items())
    #     return result

    # def corpus2csc(corpus, num_terms=None, dtype=np.float64, num_docs=None, num_nnz=None, printprogress=0):
    #     try:
    #         # if the input corpus has the `num_nnz`, `num_docs` and `num_terms` attributes
    #         # (as is the case with MmCorpus for example), we can use a more efficient code path
    #         if num_terms is None:
    #             num_terms = corpus.num_terms
    #         if num_docs is None:
    #             num_docs = corpus.num_docs
    #         if num_nnz is None:
    #             num_nnz = corpus.num_nnz
    #     except AttributeError:
    #         pass  # not a MmCorpus...
    #     if printprogress:
    #         logger.info("creating sparse matrix from corpus")
    #     if num_terms is not None and num_docs is not None and num_nnz is not None:
    #         # faster and much more memory-friendly version of creating the sparse csc
    #         posnow, indptr = 0, [0]
    #         indices = np.empty((num_nnz,), dtype=np.int32)  # HACK assume feature ids fit in 32bit integer
    #         data = np.empty((num_nnz,), dtype=dtype)
    #         for docno, doc in enumerate(corpus):
    #             if printprogress and docno % printprogress == 0:
    #                 logger.info("PROGRESS: at document #%i/%i", docno, num_docs)
    #             posnext = posnow + len(doc)
    #             # zip(*doc) transforms doc to (token_indices, token_counts]
    #             indices[posnow: posnext], data[posnow: posnext] = zip(*doc) if doc else ([], [])
    #             indptr.append(posnext)
    #             posnow = posnext
    #         assert posnow == num_nnz, "mismatch between supplied and computed number of non-zeros"
    #         result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    #     else:
    #         # slower version; determine the sparse matrix parameters during iteration
    #         num_nnz, data, indices, indptr = 0, [], [], [0]
    #         for docno, doc in enumerate(corpus):
    #             if printprogress and docno % printprogress == 0:
    #                 logger.info("PROGRESS: at document #%i", docno)

    #             # zip(*doc) transforms doc to (token_indices, token_counts]
    #             doc_indices, doc_data = zip(*doc) if doc else ([], [])
    #             indices.extend(doc_indices)
    #             data.extend(doc_data)
    #             num_nnz += len(doc)
    #             indptr.append(num_nnz)
    #         if num_terms is None:
    #             num_terms = max(indices) + 1 if indices else 0
    #         num_docs = len(indptr) - 1
    #         # now num_docs, num_terms and num_nnz contain the correct values
    #         data = np.asarray(data, dtype=dtype)
    #         indices = np.asarray(indices)
    #         result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    #     return result
