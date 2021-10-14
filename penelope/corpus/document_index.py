from __future__ import annotations

import logging
import os
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from loguru import logger
from penelope.utility import (
    FilenameFieldSpecs,
    PD_PoS_tag_groups,
    deprecated,
    dict_of_key_values_inverted_to_dict_of_value_key,
    extract_filenames_metadata,
    is_strictly_increasing,
    list_of_dicts_to_dict_of_lists,
    probe_extension,
    strip_path_and_extension,
    strip_paths,
)

if TYPE_CHECKING:
    from .readers.interfaces import TextReaderOpts


class DocumentIndexError(ValueError):
    ...


T = TypeVar("T", int, str)

DOCUMENT_INDEX_COUNT_COLUMNS = ["n_raw_tokens", "n_tokens"] + PD_PoS_tag_groups.index.tolist()

DocumentIndex = pd.core.api.DataFrame

# pylint: disable=too-many-public-methods


class DocumentIndexHelper:
    def __init__(self, document_index: Union[DocumentIndex, List[dict], str], **kwargs):

        if not isinstance(document_index, (DocumentIndex, list, str)):
            raise DocumentIndexError("expected document index data but found None")

        self._document_index: DocumentIndex = (
            document_index
            if isinstance(document_index, DocumentIndex)
            else metadata_to_document_index(metadata=document_index, document_id_field=None)
            if isinstance(document_index, list)
            else load_document_index_from_str(data_str=document_index, sep=kwargs.get('sep', '\t'))
        )

    @property
    def document_index(self) -> DocumentIndex:
        return self._document_index

    def store(self, filename: str) -> "DocumentIndexHelper":
        store_document_index(self.document_index, filename)
        return self

    @staticmethod
    def load(
        filename: Union[str, StringIO], *, sep: str = '\t', document_id_field: str = 'document_id', **read_csv_kwargs
    ) -> "DocumentIndexHelper":
        _index = load_document_index(filename, sep=sep, document_id_field=document_id_field, **read_csv_kwargs)
        return DocumentIndexHelper(_index)

    @staticmethod
    def from_metadata(metadata: List[Dict], *, document_id_field: str = None) -> "DocumentIndexHelper":
        _index = metadata_to_document_index(metadata, document_id_field=document_id_field)
        return DocumentIndexHelper(_index)

    @staticmethod
    def from_filenames(filenames: List[str], filename_fields: FilenameFieldSpecs) -> "DocumentIndexHelper":

        if filename_fields is None:
            return None

        if hasattr(filename_fields, 'filename_fields'):
            """Is actually a TextReaderOpts"""
            filename_fields = filename_fields.filename_fields

        _metadata = extract_filenames_metadata(filenames=filenames, filename_fields=filename_fields)
        _index = metadata_to_document_index(_metadata)

        return DocumentIndexHelper(_index)

    @staticmethod
    def from_filenames2(filenames: List[str], reader_opts: TextReaderOpts) -> Optional[DocumentIndex]:

        if not reader_opts or reader_opts.filename_fields is None:
            return None

        _index: DocumentIndex = DocumentIndexHelper.from_filenames(
            filenames=filenames, filename_fields=reader_opts.filename_fields
        ).document_index

        return _index

    @staticmethod
    def from_str(data_str: str, sep: str = '\t', document_id_field: str = 'document_id') -> "DocumentIndexHelper":
        _index = load_document_index_from_str(data_str=data_str, sep=sep, document_id_field=document_id_field)
        return DocumentIndexHelper(_index)

    def consolidate(self, reader_index: DocumentIndex) -> "DocumentIndexHelper":
        self._document_index = consolidate_document_index(self._document_index, reader_index)
        return self

    @deprecated
    def upgrade(self) -> "DocumentIndexHelper":
        self._document_index = document_index_upgrade(self._document_index)
        return self

    def update_counts(self, doc_token_counts: List[Tuple[str, int, int]]) -> "DocumentIndexHelper":
        self._document_index = update_document_index_token_counts(
            self._document_index, doc_token_counts=doc_token_counts
        )
        return self

    def update_counts_by_corpus(self, corpus: Any, column_name: str = 'n_terms') -> "DocumentIndexHelper":
        self._document_index = update_document_index_token_counts_by_corpus(
            self._document_index, corpus=corpus, column_name=column_name
        )
        return self

    def add_attributes(self, other: DocumentIndex) -> "DocumentIndexHelper":
        """ Adds other's document meta data (must have a document_id) """
        self._document_index = self._document_index.merge(
            other, how='inner', left_on='document_id', right_on='document_id'
        )
        return self

    def update_properties(self, *, document_name: str, property_bag: Mapping[str, int]) -> "DocumentIndexHelper":
        """Updates attributes for the specified document item"""
        # property_bag: dict = {k: property_bag[k] for k in property_bag if k not in ['document_name']}
        # for key in [k for k in property_bag if k not in self._document_index.columns]:
        #     self._document_index.insert(len(self._document_index.columns), key, np.nan)
        # self._document_index.update(DocumentIndex(data=property_bag, index=[document_name], dtype=np.int64))
        update_document_index_properties(self._document_index, document_name=document_name, property_bag=property_bag)
        return self

    def overload(self, df: DocumentIndex, column_names: List[str]) -> DocumentIndex:
        return overload_by_document_index_properties(self._document_index, df=df, column_names=column_names)

    def apply_filename_fields(self, filename_fields: FilenameFieldSpecs):
        apply_filename_fields(self._document_index, filename_fields=filename_fields)

    def group_by_column(
        self,
        pivot_column_name: str,
        transformer: Union[Callable[[T], T], Dict[T, T], None] = None,
        index_values: Union[str, List[T]] = None,
        extra_grouping_columns: List[str] = None,
        target_column_name: str = 'category',
    ) -> "DocumentIndexHelper":
        """Returns a reduced document index grouped by specified column.

        If `transformer` is None then grouping is done on `column_name`.
        Otherwise a new `target_column_name` column is added by applying `transformer` to `pivot_column_name`.

        All columns found in COUNT_COLUMNS will be summed up.

        New `filename`, `document_name` and `document_id` columns are generated.
        Both `filename` and `document_name` will be set to str(`category`)

        If `index_values` is specified than the returned index will have _exactly_ those index
        values, and can be used for instance if there are gaps in the index that needs to be filled.

        For integer categories `index_values` can have the literal value `fill_gaps` in which case
        a strictly increasing index will be created without gaps.

        Args:
            pivot_column_name (str): The column to group by, must exist, must be of int or str type
            transformer (callable, dict, None): Transforms to apply to column before grouping
            index_values (pd.Series, List[T]): pandas index of returned document index
            target_column_name (str): Name of resulting category column

        Raises:
            DocumentIndexError: [description]

        Returns:
            [type]: [description]
        """

        if extra_grouping_columns:
            raise NotImplementedError("Use of extra_grouping_columns is NOT implemented")

        if pivot_column_name not in self._document_index.columns:
            raise DocumentIndexError(f"fatal: document index has no {pivot_column_name} column")

        """
        Create `agg` dict that sums up all count variables (and span of years per group)
        Adds or updates n_documents column. Sums up `n_documents` if it exists, other counts distinct `document_id`
        """
        count_aggregates = {
            **{
                count_column: 'sum'
                for count_column in DOCUMENT_INDEX_COUNT_COLUMNS
                if count_column in self._document_index.columns
            },
            **({} if pivot_column_name == 'year' else {'year': ['min', 'max', 'size']}),
            **(
                {'document_id': 'nunique'}
                if "n_documents" not in self._document_index.columns
                else {'n_documents': 'sum'}
            ),
        }

        transform = lambda df: (
            df[pivot_column_name]
            if transformer is None
            else df[pivot_column_name].apply(transformer)
            if callable(transformer)
            else df[pivot_column_name].apply(transformer.get)
            if isinstance(transformer, dict)
            else None
        )

        document_index: DocumentIndex = (
            self._document_index.assign(**{target_column_name: transform})
            .groupby([target_column_name])
            .agg(count_aggregates)
        )

        # Reset column index to a single level
        document_index.columns = [col if isinstance(col, str) else '_'.join(col) for col in document_index.columns]

        document_index = document_index.rename(
            columns={
                'document_id': 'n_documents',
                'document_id_nunique': 'n_documents',
                'n_documents_sum': 'n_documents',
                'n_raw_tokens_sum': 'n_raw_tokens',
                'year_size': 'n_years',
            }
        )

        # Set new index `index_values` as new index if specified, or else index
        if index_values is None:

            # Use existing index values (results from group by)
            index_values = document_index.index

        elif isinstance(index_values, str) and index_values == 'fill_gaps':
            # Create a strictly increasing index (fills gaps, index must be of integer type)

            if not np.issubdtype(document_index.dtype, np.integer):
                raise DocumentIndexError(f"expected index of type int, found {type(document_index.dtype)}")

            index_values = np.arange(document_index.index.min(), document_index.index.max() + 1, 1)

        # Create new data frame with given index values, add columns and left join with grouped index
        document_index = pd.merge(
            pd.DataFrame(
                {
                    target_column_name: index_values,
                    'filename': [f"{pivot_column_name}_{value}.txt" for value in index_values],
                    'document_name': [f"{pivot_column_name}_{value}" for value in index_values],
                }
            ).set_index(target_column_name),
            document_index,
            how='left',
            left_index=True,
            right_index=True,
        )

        # Add `year` column if grouping column was 'year`, set to index or min year based on grouping column
        document_index['year'] = document_index.index if pivot_column_name == 'year' else document_index.year_min

        # Add `document_id`
        document_index = document_index.reset_index(drop=document_index.index.name in document_index.columns)
        document_index['document_id'] = document_index.index

        # Set `document_name` as index of result data frame
        document_index = document_index.set_index('document_name', drop=False).rename_axis('')

        return DocumentIndexHelper(document_index)

    def group_by_time_period(
        self,
        *,
        time_period_specifier: Union[str, dict, Callable[[Any], Any]],
        source_column_name: str = 'year',
        target_column_name: str = 'time_period',
        index_values: Union[str, List[T]] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """Special case of of above, groups by 'year' based on `time_period_specifier`

            time_period_specifier specifies transform of source_column_name to target_column_name prior to grouping:
               - If Callable then it is applied on source_column_name
               - If literal 'decade' or 'lustrum' then source column is assumed to be a year


        Args:
            time_period_specifier (Union[str, dict, Callable[[Any], Any]]): Group category specifier
            target_column_name (str, optional): Category column name. Defaults to 'time_period'.

        Returns:
            Tuple[pd.DataFrame, dict]: grouped document index and group indices
        """

        """Add new column `target_column_name`"""
        self._document_index[target_column_name] = (
            self._document_index[source_column_name]
            if time_period_specifier == source_column_name
            else self._document_index.year.apply(create_time_period_categorizer(time_period_specifier))
        )

        """Store indices for documents in each group"""
        category_indices = self._document_index.groupby(target_column_name).apply(lambda x: x.index.tolist()).to_dict()

        """Group by 'target_column_name' column"""
        grouped_document_index = (
            self.group_by_column(
                pivot_column_name=target_column_name,
                extra_grouping_columns=None,
                target_column_name=target_column_name,
                index_values=index_values,
            )
            .document_index.set_index('document_id', drop=False)
            .sort_index(axis=0)
        )

        """Fix result names"""
        grouped_document_index.columns = [name.replace('_sum', '') for name in grouped_document_index.columns]

        return grouped_document_index, category_indices

    def set_strictly_increasing_index(self) -> "DocumentIndexHelper":
        """Sets a strictly increasing index"""
        self._document_index['document_id'] = get_strictly_increasing_document_id(
            self._document_index, document_id_field=None
        )
        self._document_index = self._document_index.set_index('document_id', drop=False).rename_axis('')
        return self

    def extend(self, other_index: DocumentIndex) -> "DocumentIndexHelper":
        if self._document_index is None:
            self._document_index = other_index
        else:
            # if not self._document_index.columns.equals(self._document_index.columns):
            #     raise ValueError("Document index columns mismatch")
            self._document_index = self._document_index.append(other_index, ignore_index=False)
            self._document_index['document_id'] = range(0, len(self._document_index))
        return self

    @staticmethod
    def year_range(document_index: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
        """Returns document's year range

        Returns
        -------
        Tuple[Optional[int],Optional[int]]
            Min/max document year
        """
        if 'year' in document_index.columns:
            return (document_index.year.min(), document_index.year.max())
        return (None, None)

    @staticmethod
    def xs_years(document_index: pd.DataFrame) -> Tuple[int, int]:
        """Returns an array that contains a no-gap year sequence from min year to max year

        Returns
        -------
        numpy.array
            Sequence from min year to max year
        """
        (low, high) = DocumentIndexHelper.year_range(document_index)
        xs = np.arange(low, high + 1, 1)
        return xs


KNOWN_TIME_PERIODS: dict = {'year': 1, 'lustrum': 5, 'decade': 10}

TimePeriodSpecifier = Union[str, dict, Callable[[Any], Any]]


def get_document_id(document_index: DocumentIndex, document_name: str) -> int:
    document_id = document_index.loc[document_name]['document_id']
    return document_id


def create_time_period_categorizer(time_period_specifier: TimePeriodSpecifier) -> Callable[[Any], Any]:
    # FIXME: Move to pandas_utils or time_period_utils.py

    if callable(time_period_specifier):
        return time_period_specifier

    if isinstance(time_period_specifier, str):

        if time_period_specifier not in KNOWN_TIME_PERIODS:
            raise ValueError(f"{time_period_specifier} is not a known period specifier")

        categorizer = lambda y: y - int(y % KNOWN_TIME_PERIODS[time_period_specifier])

    else:

        year_group_mapping = dict_of_key_values_inverted_to_dict_of_value_key(time_period_specifier)

        categorizer = lambda x: year_group_mapping.get(x, np.nan)

    return categorizer


def get_strictly_increasing_document_id(
    document_index: DocumentIndex, document_id_field: str = 'document_id'
) -> pd.Series:
    """[summary]

    Args:
        document_index (DocumentIndex): [description]
        document_id_field (str): [description]

    Returns:
        pd.Series: [description]
    """

    if document_id_field in document_index.columns:
        if is_strictly_increasing(document_index[document_id_field]):
            return document_index[document_id_field]

    if is_strictly_increasing(document_index.index):
        return document_index.index

    if document_index.index.dtype == np.dtype('int64'):
        # Logic from deprecated document_index_upgrade() should never happen
        raise ValueError("Integer index encountered that are not strictly increasing!")
        # if 'document_id' not in document_index.columns:
        #     document_index['document_id'] = document_index.index

    return document_index.reset_index().index


def store_document_index(document_index: DocumentIndex, filename: str) -> None:
    """[summary]

    Args:
        document_index (DocumentIndex): [description]
        filename (str): [description]
    """

    compression: dict = dict(method='zip', archive_name="document_index.csv") if filename.endswith('zip') else 'infer'

    document_index.to_csv(filename, sep='\t', compression=compression, header=True)


def load_document_index(
    filename: Union[str, StringIO, DocumentIndex],
    *,
    sep: str,
    document_id_field: str = 'document_id',
    filename_fields: FilenameFieldSpecs = None,
    probe_extensions: str = 'zip,csv,gz',
    **read_csv_kwargs,
) -> DocumentIndex:
    """Loads a document index and sets `document_name` as index column. Also adds `document_id` if missing"""

    if filename is None:
        return None

    if isinstance(filename, DocumentIndex):
        document_index: DocumentIndex = filename
    else:
        if isinstance(filename, str):
            if (filename := probe_extension(filename, extensions=probe_extensions)) is None:
                raise FileNotFoundError(f"{filename} (probed: {probe_extensions})")

        document_index: DocumentIndex = pd.read_csv(filename, sep=sep, **read_csv_kwargs)

    for old_or_unnamed_index_column in ['Unnamed: 0', 'filename.1']:
        if old_or_unnamed_index_column in document_index.columns:
            document_index = document_index.drop(old_or_unnamed_index_column, axis=1)

    if 'filename' not in document_index.columns:
        raise DocumentIndexError("expected mandatory column `filename` in document index, found no such thing")

    document_index['document_id'] = get_strictly_increasing_document_id(document_index, document_id_field).astype(
        np.int32
    )

    if 'document_name' not in document_index.columns or (document_index.document_name == document_index.filename).all():
        document_index['document_name'] = document_index.filename.apply(strip_path_and_extension)

    document_index = document_index.set_index('document_name', drop=False).rename_axis('')

    if filename_fields is not None:
        document_index = apply_filename_fields(document_index, filename_fields)

    if 'year' in document_index:
        document_index['year'] = document_index.year.astype(np.int16)

    return document_index


@deprecated
def document_index_upgrade(document_index: DocumentIndex) -> DocumentIndex:
    """Fixes older versions of document indexes"""

    if 'document_name' not in document_index.columns:
        document_index['document_name'] = document_index.filename.apply(strip_path_and_extension)

    if document_index.index.dtype == np.dtype('int64'):

        if 'document_id' not in document_index.columns:
            document_index['document_id'] = document_index.index

    document_index = document_index.set_index('document_name', drop=False).rename_axis('')

    return document_index


def metadata_to_document_index(metadata: List[Dict], *, document_id_field: str = 'document_id') -> DocumentIndex:
    """Creates a document index from collected filename fields metadata."""

    if metadata is None or len(metadata) == 0:
        metadata = {'filename': [], 'document_id': []}

    document_index = load_document_index(DocumentIndex(metadata), sep=None, document_id_field=document_id_field)

    return document_index


def apply_filename_fields(document_index: DocumentIndex, filename_fields: FilenameFieldSpecs):
    """Extends document index with filename fields defined by `filename_fields`"""
    if 'filename' not in document_index.columns:
        raise DocumentIndexError("filename not in document index")
    filenames = [strip_paths(filename) for filename in document_index.filename.tolist()]
    metadata: List[Mapping[str, Any]] = extract_filenames_metadata(filenames=filenames, filename_fields=filename_fields)
    for key, values in list_of_dicts_to_dict_of_lists(metadata).items():
        if key not in document_index.columns:
            document_index[key] = values
    return document_index


def load_document_index_from_str(data_str: str, sep: str, document_id_field: str = 'document_id') -> DocumentIndex:
    df = load_document_index(StringIO(data_str), sep=sep, document_id_field=document_id_field)
    return df


def consolidate_document_index(document_index: DocumentIndex, reader_index: DocumentIndex) -> DocumentIndex:
    """Returns a consolidated document index from an existing index, if exists,
    and the reader index."""

    if document_index is not None:
        columns = [x for x in reader_index.columns if x not in document_index.columns]
        if len(columns) > 0:
            document_index = document_index.merge(reader_index[columns], left_index=True, right_index=True, how='left')
        return document_index

    return reader_index


def update_document_index_token_counts(
    document_index: DocumentIndex, doc_token_counts: List[Tuple[str, int, int]]
) -> DocumentIndex:
    """Updates or adds fields `n_raw_tokens` and `n_tokens` to document index from collected during a corpus read pass
    Only updates values that don't already exist in the document index"""
    try:

        strip_ext = lambda filename: os.path.splitext(filename)[0]

        df_counts: pd.DataFrame = pd.DataFrame(data=doc_token_counts, columns=['filename', 'n_raw_tokens', 'n_tokens'])
        df_counts['document_name'] = df_counts.filename.apply(strip_ext)
        df_counts = df_counts.set_index('document_name').rename_axis('').drop('filename', axis=1)

        if 'document_name' not in document_index.columns:
            document_index['document_name'] = document_index.filename.apply(strip_ext)

        if 'n_raw_tokens' not in document_index.columns:
            document_index['n_raw_tokens'] = np.nan

        if 'n_tokens' not in document_index.columns:
            document_index['n_tokens'] = np.nan

        document_index.update(df_counts)

    except Exception as ex:
        logging.error(ex)

    return document_index


def update_document_index_token_counts_by_corpus(
    document_index: pd.DataFrame, corpus: Any, column_name: str = 'n_terms'
) -> pd.DataFrame:
    """Variant used in topic modeling"""
    if column_name in document_index.columns:
        return document_index

    n_terms: List[int] = None

    try:

        if hasattr(corpus, 'sparse'):
            # Gensim Sparse2Corpus
            # FIXME: Kolla att detta är rätt! Ska det vara axis=0???
            n_terms = corpus.sparse.sum(axis=0).A1
        elif hasattr(corpus, 'data'):
            # Vectorized corpus
            n_terms = corpus.document_token_counts
        elif isinstance(corpus, list):
            # BoW, men hur vara säker?
            n_terms = [sum((w[1] for w in d)) for d in corpus]
        else:
            n_terms = [len(d) for d in corpus]

    except Exception as ex:
        logger.exception(ex)

    if n_terms is not None:
        document_index[column_name] = n_terms

    return document_index


def update_document_index_properties(
    document_index: DocumentIndex,
    *,
    document_name: str,
    property_bag: Mapping[str, int],
) -> None:
    """[summary]

    Args:
        document_index ([type]): [description]
        document_name (str): [description]
        property_bag (Mapping[str, int]): [description]
    """
    if 'document_name' in property_bag:
        property_bag: dict = {k: v for k, v in property_bag.items() if k != 'document_name'}

    for key in [k for k in property_bag if k not in document_index.columns]:
        document_index.insert(len(document_index.columns), key, np.nan)

    document_index.loc[document_name, property_bag.keys()] = property_bag.values()
    # document_index.update(pd.DataFrame(data=property_bag, index=[document_name], dtype=np.int64))


def update_document_index_key_values(
    document_index: DocumentIndex,
    key_column_name: str,
    key_value_bag: dict,
    default_value: Any = np.nan,  # Mapping[str, Any]
) -> DocumentIndex:
    """Updates column with values found in key_value_bag dictionary (keys are index, and value is column value). Creates column if missing."""
    df_property_bag: pd.DataFrame = pd.DataFrame.from_dict(key_value_bag, orient='index').rename(
        {0: key_column_name}, axis=1
    )
    if len(df_property_bag) == len(document_index):
        document_index[key_column_name] = df_property_bag[key_column_name]
    else:
        if key_column_name not in document_index.columns:
            document_index[key_column_name] = default_value
        document_index.update(df_property_bag)
    return document_index


def overload_by_document_index_properties(
    document_index: DocumentIndex, df: pd.DataFrame, column_names: List[str] = None
) -> DocumentIndex:
    """Add document `columns` to `df` if columns not already exists.

    Parameters
    ----------
    document_index : DocumentIndex
        Corpus document index, by default None
    df : pd.DataFrame
        Data of interest
    columns : Union[str,List[str]]
        Columns in `document_index` that should be added to `df`

    Returns
    -------
    DocumentIndex
        `df` extended with `columns` data
    """

    if column_names is None:
        column_names = document_index.columns.tolist()

    if document_index is None:
        return df

    if 'document_id' not in df.columns:
        return df

    if isinstance(column_names, str):
        column_names = [column_names]

    column_names = ['document_id'] + [c for c in column_names if c not in df.columns and c in document_index.columns]

    if len(column_names) == 1:
        return df

    overload_data: pd.DataFrame = document_index[column_names].set_index('document_id')

    df = df.merge(overload_data, how='inner', left_on='document_id', right_index=True)

    return df


def count_documents_in_index_by_pivot(document_index: DocumentIndex, attribute: str) -> List[int]:
    """Return a list of document counts per group defined by attribute
    Assumes documents are sorted by attribute!
    """
    assert document_index[attribute].is_monotonic_increasing, 'Documents *MUST* be sorted by TIME-SLICE attribute!'
    # TODO: Either sort documents (and corpus or term stream!) prior to this call - OR force sortorder by filename (add year to filename)
    return list(document_index.groupby(attribute).size().values)
