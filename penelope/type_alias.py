from typing import Callable, Iterable, List, Tuple, Union

import pandas as pd

PartitionKeys = Union[str, List[str], Callable]
FilenameTokensTuple = Tuple[str, Iterable[str]]
FilenameTokensTuples = Iterable[FilenameTokensTuple]
CoOccurrenceDataFrame = pd.DataFrame
DocumentIndex = pd.DataFrame
