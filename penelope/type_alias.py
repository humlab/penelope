from typing import Callable, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import pandas as pd
import scipy.sparse as sp

TokenWeights = list[tuple[str, float]]

StrBoW = list[tuple[str, float]]
IntBoW = list[tuple[int, float]]

PartitionKeys = Union[str, List[str], Callable]
FilenameTokensTuple = Tuple[str, Iterable[str]]
FilenameTokensTuples = Iterable[FilenameTokensTuple]

DocumentIndex = pd.DataFrame
IntOrStr = Union[int, str]

TaggedFrame = pd.DataFrame
Token = Union[int, str]

DocumentTermsStream = Iterable[Tuple[str, Iterable[str]]]

WindowsStream = Iterator[Tuple[str, int, Iterable[Token]]]
CoOccurrenceDataFrame = pd.DataFrame
VocabularyMapping = Optional[Mapping[Tuple[int, int], int]]
SparseMatrix = sp.spmatrix

SerializableContent = Union[str, Iterable[str], TaggedFrame]

DocumentTopicsWeightsIter = Iterable[Tuple[int, Iterable[Tuple[int, float]]]]
