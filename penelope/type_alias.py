from typing import Callable, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import pandas as pd
import scipy.sparse as sp

PartitionKeys = Union[str, List[str], Callable]
FilenameTokensTuple = Tuple[str, Iterable[str]]
FilenameTokensTuples = Iterable[FilenameTokensTuple]

DocumentIndex = pd.DataFrame
IntOrStr = Union[int, str]

TaggedFrame = pd.core.api.DataFrame
Token = Union[int, str]

WindowsStream = Iterator[Tuple[str, int, Iterable[Token]]]
CoOccurrenceDataFrame = pd.DataFrame
VocabularyMapping = Optional[Mapping[Tuple[int, int], int]]
SparseMatrix = sp.spmatrix

SerializableContent = Union[str, Iterable[str], TaggedFrame]
