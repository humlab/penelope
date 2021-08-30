import pathlib
import zipfile
from collections import defaultdict
from collections.abc import MutableMapping
from fnmatch import fnmatch
from typing import Any, Callable, Container, Iterable, Iterator, List, Mapping, Optional, Set, Tuple, Union

import pandas as pd
from loguru import logger
from penelope.corpus.readers import GLOBAL_TF_THRESHOLD_MASK_TOKEN
from penelope.utility import path_add_suffix, pickle_to_file, replace_extension, strip_paths, unpickle_from_file

MAGIC_TOKENS = {"*", GLOBAL_TF_THRESHOLD_MASK_TOKEN}

# pylint: disable=too-many-public-methods


class ClosedVocabularyError(Exception):
    ...


class Token2Id(MutableMapping):
    """A token-to-id mapping (dictionary)"""

    def __init__(
        self, data: Optional[Union[dict, defaultdict]] = None, tf: dict = None, fallback_token: str = None, **kwargs
    ):

        self._data: defaultdict = None
        self._tf: dict = None
        self._is_open = True
        self._id2token: dict = None
        self._fallback_token: str = fallback_token
        self._payload: dict = dict(**kwargs)

        self.replace(data=data or defaultdict(), tf=tf)

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        if not self._is_open:
            if self._fallback_token:
                return self._data.get(key, self._fallback_token)
        return self._data[key]

    def __optimized__getitem__(self) -> Callable[[str], int]:
        """Optimises __getitem__ by wireing up a replacement closure without conditional constructs"""
        data = self._data
        if self._is_open:
            return lambda w: data[w]
        fallback_token = self._fallback_token
        if fallback_token is None:
            return lambda w: data[w]
        return lambda key: data.get(key, fallback_token)

    def __setitem__(self, key: str, value):
        if self._id2token:
            self._id2token = None
        if not self.is_open:
            raise ClosedVocabularyError(f"cannot add item to a closed vocabulary: '{value}'")
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @property
    def data(self) -> defaultdict:
        return self._data

    @property
    def tf(self) -> dict:
        return self._tf

    @property
    def magic_tokens(self) -> List[str]:
        return MAGIC_TOKENS

    @property
    def magic_token_ids(self) -> List[str]:
        return [self[w] for w in MAGIC_TOKENS if w in self._data]

    # @data.setter
    # def data(self, value: Any) -> None:
    #     self.replace(value)

    def replace(self, *, data: Any, tf: dict = None) -> "Token2Id":
        """Replace current data with `data`"""
        if isinstance(data, defaultdict):
            self._data = data
        elif isinstance(data, dict):
            self._data = defaultdict(int, data)
        else:
            self._data = data or defaultdict()
        self._id2token = None
        if tf is not None:
            self._tf = tf
        self.sync_state()
        return self

    @property
    def payload(self) -> Mapping[Any, Any]:
        return self._payload

    def remember(self, **kwargs) -> "Token2Id":
        """Stores items in payload"""
        self.payload.update(kwargs)
        return self

    def recall(self, key: str) -> Optional[Any]:
        """Retrieves item from payload"""
        return self.payload.get(key)

    def ingest(self, tokens: Iterator[str]) -> "Token2Id":

        if not self._is_open:
            raise ClosedVocabularyError("cannot ingest into a closed vocabulary")

        if self._tf is None:
            self._tf = defaultdict(int)

        self._id2token = None

        data = self._data
        tf = self._tf

        for t in tokens:
            tf[data[t]] += 1

        return self

    def ingest_stream(self, tokens_stream: Iterator[Union[Iterator[str], dict]]) -> "Token2Id":
        if not self._is_open:
            raise ClosedVocabularyError("cannot ingest into a closed vocabulary")

        if self._tf is None:
            self._tf = defaultdict(int)

        self._id2token = None

        self._ingest_stream(tokens_stream)
        # self._tf.update(data[t] for tokens in tokens_stream for t in tokens )
        return self

    def _ingest_stream(self, tokens_stream: Iterator[Iterator[str]]) -> None:

        tf: defaultdict = self._tf
        data = self._data

        for d in tokens_stream:
            if isinstance(d, dict):
                for t, v in d.items():
                    tf[data[t]] += v
            else:
                for t in d:
                    tf[data[t]] += 1

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def fallback_token(self) -> str:
        return self._fallback_token

    @fallback_token.setter
    def fallback_token(self, value: str) -> None:
        self._fallback_token = value
        self.__getitem__ = self.__optimized__getitem__()

    def close(self, fallback: int = None) -> "Token2Id":
        self._data.default_factory = None
        self._fallback_token = fallback
        self._is_open = False
        self.__getitem__ = self.__optimized__getitem__()
        return self

    def open(self) -> "Token2Id":
        self._data.default_factory = self._data.__len__
        self._id2token = None
        self._is_open = True
        self.__getitem__ = self.__optimized__getitem__()
        return self

    def default(self, value: int) -> "Token2Id":
        self._data.default_factory = lambda: value
        self._is_open = False
        self.__getitem__ = self.__optimized__getitem__()
        return self

    @property
    def id2token(self) -> dict:
        # FIXME: Always create new reversed mapping if vocabulay is open
        if self._id2token is None or len(self) != len(self._id2token):  # or self.is_open:
            self._id2token = {v: k for k, v in self._data.items()}
        return self._id2token

    def to_dataframe(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame({'token': self._data.keys(), 'token_id': self._data.values()}).set_index(
            'token'
        )
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
        """Load vocabulary from CSV"""
        if not pathlib.Path(filename).exists():
            logger.info(f"Token2Id.load: filename {filename} not found")
            return None
        df: pd.DataFrame = pd.read_csv(filename, sep='\t', index_col=0, na_filter=False)
        data: dict = df['token_id'].to_dict()
        tf: defaultdict = Token2Id.load_tf(filename)
        token2id: Token2Id = Token2Id(data=data, tf=tf)
        return token2id

    @staticmethod
    def load_tf(filename: str) -> Optional[dict]:
        tf_filename: str = path_add_suffix(filename, "_tf", new_extension=".pbz2")
        tf: Any = unpickle_from_file(tf_filename) if pathlib.Path(tf_filename).exists() else None
        if not isinstance(tf, defaultdict):
            tf = defaultdict(int, tf)
        return tf

    def store_tf(self, filename: str) -> None:
        if not self._tf:
            return
        tf_filename: str = path_add_suffix(filename, "_tf", new_extension=".pbz2")
        pickle_to_file(tf_filename, self._tf)

    def to_ids(self, tokens: List[str]) -> List[int]:
        return [self._data[w] for w in tokens]

    def to_id_set(self, tokens: Iterable[str]) -> Set[int]:
        return {self._data[w] for w in tokens}

    def find(self, what: Union[List[str], str]):

        if not what:
            return []

        if isinstance(what, (int, str)):
            what = [what]

        wildcards = [w for w in what if '*' in w]
        tokens = [w for w in what if w not in wildcards]

        matches = []

        if tokens:
            matches.extend([w for w in tokens if w in self._data])

        if wildcards:
            matches.extend([w for w in self._data.keys() if any(fnmatch(w, x) for x in wildcards)])

        return [self[w] for w in set(matches)]

    def compress(
        self, *, tf_threshold: int = 1, inplace=False, keeps: Container[Union[int, str]] = None
    ) -> Tuple["Token2Id", Mapping[int, int]]:
        """Returns a compressed version of corpus, with ID translation, where tokens below threshold are removed"""

        if tf_threshold <= 1:
            return self, None

        keeps: Container[int] = {self[x] if isinstance(x, str) else x for x in keeps} if keeps else set()
        keeps |= set(self.magic_token_ids)

        logger.info(
            f"Compressing vocab: TF-threshold={tf_threshold} Keeping: {' '.join([self.id2token[x] for x in keeps])}"
        )

        if self.tf is None:
            raise ValueError("Token2Id.compress: cannot compress when TF counts is none!")

        tf: dict = self.tf

        """Create translation between old IDs and new IDs"""

        translation: Mapping[int, int] = {
            old_token_id: (new_token_id, v)
            for new_token_id, (old_token_id, v) in enumerate(
                (k, v) for (k, v) in tf.items() if (v >= tf_threshold or k in keeps)
            )
        }

        new_tf: dict = defaultdict(int, {k: v for (k, v) in translation.values()})
        new_data = {
            w: translation[old_token_id][0] for (w, old_token_id) in self._data.items() if old_token_id in translation
        }

        """Add and sum-up masked low-tf marker"""

        if GLOBAL_TF_THRESHOLD_MASK_TOKEN not in new_data:
            new_data[GLOBAL_TF_THRESHOLD_MASK_TOKEN] = len(new_data)

        old_mask_id = self._data[GLOBAL_TF_THRESHOLD_MASK_TOKEN] if GLOBAL_TF_THRESHOLD_MASK_TOKEN in self else -1
        mask_id = new_data[GLOBAL_TF_THRESHOLD_MASK_TOKEN]

        if mask_id not in new_tf:
            new_tf[mask_id] = 0

        new_tf[mask_id] += sum(tf[i] for i in self._data.values() if i not in translation and i != old_mask_id)

        ids_translation = {k: v[0] for k, v in translation.items()}
        if inplace:
            self.replace(data=new_data, tf=new_tf)
            return self.close(fallback=mask_id), ids_translation

        token2id: Token2Id = Token2Id(data=new_data, tf=new_tf).close(fallback=mask_id)

        return token2id, ids_translation

    # @deprecated
    # def clip(self, keep_ids: List[int], inplace: bool = True) -> "Token2Id":
    #     """Removes tokens not found in `keep_ids` """
    #     keep_ids = set(keep_ids)
    #     dg, tg = self.id2token.get, self._tf.get
    #     data = defaultdict(None, dict({dg(token_id): token_id for token_id in keep_ids}))
    #     tf = defaultdict(int, {token_id: tg(token_id) for token_id in keep_ids})

    #     if inplace:
    #         return self.replace(data=data, tf=tf)

    #     token2id = Token2Id(data=data, tf=tf, fallback_token=self.fallback_token).sync_state(self.is_open)

    #     return token2id

    def translate(self, ids_translation: Mapping[int, int], inplace: bool = True) -> "Token2Id":
        """Translates ID in vocabulary according to mapping specified in `vocab_translation`
        Translation is a mapping from old ID to new ID.
        Old item IDs that don't exist in translation are filtered out.
        """
        data = defaultdict(None, {w: ids_translation[oid] for w, oid in self._data.items() if oid in ids_translation})

        cg = self.tf.get
        tf = defaultdict(int, {ids_translation[oid]: cg(oid, 0) for oid in ids_translation})

        if inplace:
            return self.replace(data=data, tf=tf)

        token2id = Token2Id(data=data, tf=tf, fallback_token=self.fallback_token).sync_state(self.is_open)

        return token2id

    def sync_state(self, is_open: bool = None) -> "Token2Id":
        is_open = self.is_open if is_open is None else is_open
        if is_open:
            self.open()
        else:
            self.close()
        return self
