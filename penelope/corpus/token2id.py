import pathlib
import zipfile
from collections import Counter, defaultdict
from collections.abc import MutableMapping
from fnmatch import fnmatch
from typing import Container, Iterator, List, Mapping, Optional, Union

import pandas as pd
from loguru import logger
from penelope.corpus.readers import GLOBAL_TF_THRESHOLD_MASK_TOKEN
from penelope.utility import path_add_suffix, pickle_to_file, replace_extension, strip_paths, unpickle_from_file


class ClosedVocabularyError(Exception):
    ...


class Token2Id(MutableMapping):
    """A token-to-id mapping (dictionary)"""

    def __init__(self, data: Optional[Union[dict, defaultdict]] = None, tf: Counter = None):
        if isinstance(data, defaultdict):
            self.data = data
        elif isinstance(data, dict):
            self.data = defaultdict(int, data)
        else:
            self.data = data or defaultdict()
        self.tf: Counter = tf
        self._id2token: dict = None
        self.fallback_token = None
        self.data.default_factory = self.data.__len__
        self._is_open = True

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        if not self._is_open:
            if self.fallback_token:
                return self.data.get(key, self.fallback_token)
        return self.data[key]

    def __setitem__(self, key: str, value):
        if self._id2token:
            self._id2token = None
        if not self.is_open:
            raise ClosedVocabularyError(f"cannot add item to a closed vocabulary: '{value}'")
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def ingest(self, tokens: Iterator[str]) -> "Token2Id":
        if not self._is_open:
            raise ClosedVocabularyError("cannot ingest into a closed vocabulary")

        if self.tf is None:
            self.tf = Counter()

        self._id2token = None
        token_ids = [self.data[t] for t in tokens]
        self.tf.update(token_ids)
        return self

    @property
    def is_open(self) -> bool:
        return self._is_open

    def close(self, fallback: int = None) -> "Token2Id":
        self.data.default_factory = None
        self.fallback_token = fallback
        self._is_open = False
        return self

    def open(self) -> "Token2Id":

        if self.is_open:
            return self

        if self.data.default_factory is not None:
            raise ValueError("Token2Id cannot be opened with current state")

        self.data.default_factory = self.data.__len__
        self._id2token = None
        self._is_open = True
        return self

    def default(self, value: int) -> "Token2Id":
        self.data.default_factory = lambda: value
        self._is_open = False
        return self

    @property
    def id2token(self) -> dict:
        # FIXME: Always create new reversed mapping if vocabulay is open
        if self._id2token is None or len(self) != len(self._id2token):  # or self.is_open:
            self._id2token = {v: k for k, v in self.data.items()}
        return self._id2token

    def to_dataframe(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame({'token': self.data.keys(), 'token_id': self.data.values()}).set_index('token')
        return df

    def store(self, filename: str) -> "Token2Id":
        """Store dictionary as CSV"""

        # pandas_to_csv_zip(filename, dfs=(self.to_dataframe(), strip_paths(filename)), sep='\t', header=True)
        with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED) as fp:
            data_str = self.to_dataframe().to_csv(sep='\t', header=True)
            fp.writestr(replace_extension(strip_paths(filename), ".csv"), data=data_str)

        self.store_tf(filename)

        return self

    @staticmethod
    def load(filename: str) -> "Token2Id":
        """Store dictionary as CSV"""
        if not pathlib.Path(filename).exists():
            logger.info(f"Token2Id.load: filename {filename} not found")
            return None
        df: pd.DataFrame = pd.read_csv(filename, sep='\t', index_col=0, na_filter=False)
        data: dict = df['token_id'].to_dict()
        tf: Counter = Token2Id.load_tf(filename)
        token2id: Token2Id = Token2Id(data=data, tf=tf)
        return token2id

    @staticmethod
    def load_tf(filename: str) -> Optional[Counter]:
        tf_filename: str = path_add_suffix(filename, "_tf", new_extension=".pbz2")
        tf: Counter = unpickle_from_file(tf_filename) if pathlib.Path(tf_filename).exists() else None
        return tf

    def store_tf(self, filename: str) -> None:
        if not self.tf:
            return
        tf_filename: str = path_add_suffix(filename, "_tf", new_extension=".pbz2")
        pickle_to_file(tf_filename, self.tf)

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

    def compress(self, *, tf_threshold: int = 1, inplace=False, keeps: Container[Union[int, str]] = None) -> "Token2Id":
        """Returns a compressed version of corpus where tokens below threshold are removed"""

        if tf_threshold <= 1:
            return self

        keeps: Container[int] = {self[x] if isinstance(x, str) else x for x in keeps} if keeps else set()

        logger.info(
            f"compressing vocab. TF threshold: {tf_threshold} keeping: {' '.join([self.id2token[x] for x in keeps])}"
        )

        if self.tf is None:
            raise ValueError("Token2Id.compress: cannot compress when TF counts is none!")

        tf: Counter = self.tf

        translation: Mapping[int, int] = {
            token_id: (i, v)
            for i, (token_id, v) in enumerate((k, v) for (k, v) in tf.items() if (v >= tf_threshold or k in keeps))
        }

        new_tf: Counter = Counter({k: v for (k, v) in translation.values()})
        new_data = {w: translation[i][0] for (w, i) in self.data.items() if i in translation}

        if GLOBAL_TF_THRESHOLD_MASK_TOKEN not in new_data:
            new_data[GLOBAL_TF_THRESHOLD_MASK_TOKEN] = len(new_data)

        mask_id = new_data[GLOBAL_TF_THRESHOLD_MASK_TOKEN]

        if mask_id not in new_tf:
            new_tf[mask_id] = 0

        new_tf[mask_id] += sum(tf[i] for i in self.data.values() if i not in translation)

        if inplace:

            self.data = defaultdict(None, new_data)
            self.tf = new_tf

            return self.close(fallback=mask_id)

        token2id: Token2Id = Token2Id(data=new_data, tf=new_tf).close(fallback=mask_id)

        return token2id
