import logging
import os
from io import StringIO
from typing import Callable, Dict, List, Mapping, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from penelope.utility import is_strictly_increasing, strip_path_and_extension
from penelope.utility.filename_fields import FilenameFieldSpecs, extract_filenames_metadata
from penelope.utility.pos_tags import PD_PoS_tag_groups


class DocumentIndexError(ValueError):
    ...


T = TypeVar("T", int, str)

DOCUMENT_INDEX_COUNT_COLUMNS = ["n_raw_tokens", "n_tokens"] + PD_PoS_tag_groups.index.tolist()


class DocumentIndex:
    def __init__(self, document_index: Union[pd.DataFrame, List[dict], str], **kwargs):

        if not isinstance(document_index, (pd.DataFrame, list, str)):
            raise DocumentIndexError("expected document index data but found None")

        self._document_index: pd.DataFrame = (
            document_index
            if isinstance(document_index, pd.DataFrame)
            else metadata_to_document_index(metadata=document_index, document_id_field=None)
            if isinstance(document_index, list)
            else load_document_index_from_str(data_str=document_index, sep=kwargs.get('sep', '\t'))
        )

    @property
    def document_index(self):
        return self._document_index

    def store(self, filename: str) -> "DocumentIndex":
        store_document_index(self.document_index, filename)
        return self

    @staticmethod
    def load(
        filename: Union[str, StringIO], *, sep: str = '\t', document_id_field: str = 'document_id'
    ) -> "DocumentIndex":
        _index = load_document_index(filename, sep=sep, document_id_field=document_id_field)
        return DocumentIndex(_index)

    @staticmethod
    def from_metadata(metadata: List[Dict], *, document_id_field: str = None) -> "DocumentIndex":
        _index = metadata_to_document_index(metadata, document_id_field=document_id_field)
        return DocumentIndex(_index)

    @staticmethod
    def from_filenames(filenames: List[str], filename_fields: FilenameFieldSpecs) -> "DocumentIndex":
        _metadata = extract_filenames_metadata(filenames=filenames, filename_fields=filename_fields)
        _index = metadata_to_document_index(_metadata)
        return DocumentIndex(_index)

    @staticmethod
    def from_str(data_str: str, sep: str = '\t', document_id_field: str = 'document_id') -> "DocumentIndex":
        _index = load_document_index_from_str(data_str=data_str, sep=sep, document_id_field=document_id_field)
        return DocumentIndex(_index)

    def consolidate(self, reader_index: pd.DataFrame) -> "DocumentIndex":
        self._document_index = consolidate_document_index(self._document_index, reader_index)
        return self

    def upgrade(self) -> "DocumentIndex":
        self._document_index = document_index_upgrade(self._document_index)
        return self

    def update_counts(self, doc_token_counts: List[Tuple[str, int, int]]) -> "DocumentIndex":
        self._document_index = update_document_index_token_counts(
            self._document_index, doc_token_counts=doc_token_counts
        )
        return self

    def add_attributes(self, other: pd.DataFrame) -> "DocumentIndex":
        """ Adds other's document meta data (must have a document_id) """
        self._document_index = self._document_index.merge(
            other, how='inner', left_on='document_id', right_on='document_id'
        )
        return self

    def update_properties(self, *, document_name: str, property_bag: Mapping[str, int]) -> "DocumentIndex":
        """Updates attributes for the specified document item"""
        property_bag = {k: property_bag[k] for k in property_bag if k not in ['document_name']}
        for key in [k for k in property_bag if k not in self._document_index.columns]:
            self._document_index.insert(len(self._document_index.columns), key, np.nan)
        self._document_index.update(pd.DataFrame(data=property_bag, index=[document_name], dtype=np.int64))
        return self

    def group_by_column(
        self,
        column_name: str = 'year',
        transformer: Union[Callable[[T], T], Dict[T, T], None] = None,
        index_values: Union[str, List[T]] = None,
    ) -> "DocumentIndex":
        """Returns a reduced document index grouped by specified column.

        A new `category` column is added that by applying `transformer` to `column_name`.
        If `transformer` is None then `category` will be same as `column_name`.
        All count columns as specified in COUNT_COLUMNS will be summed up.
        New `filename`, `document_name` and `document_id` columns are generated.
        Both `filename` and `document_name` will be set to str(`category`)
        If `index_values` is specified than the returned index will have _exactly_ those index
        values, and can be used for instance if there are gaps n the index that needs to be filled.
        For integer categories `index_values` can have the literal value `fill_gaps` in which case
        a strictly increasing index will be created without gaps.

        Args:
            column_name (str): The column to group by, must exist, must be of int or str type
            transformer (callable, dict, None): Transforms to apply to column before grouping
            index_values (pd.Series, List[T]): pandas index of returned document index

        Raises:
            DocumentIndexError: [description]

        Returns:
            [type]: [description]
        """

        # Categrory column must exist (add before call if necessary)
        if column_name not in self._document_index.columns:
            raise DocumentIndexError(f"fatal: document index has no {column_name} column")

        # Create `agg` dict that sums up all count variables (and span of years per group)
        count_aggregates = {
            column_name: 'size',
            **{
                count_column: 'sum'
                for count_column in DOCUMENT_INDEX_COUNT_COLUMNS
                if count_column in self._document_index.columns
            },
            **({} if column_name == 'year' else {'year': ['min', 'max', 'size']}),
        }

        transform = lambda df: (
            df[column_name]
            if transformer is None
            else df[column_name].apply(transformer)
            if callable(transformer)
            else df[column_name].apply(transformer.get)
            if isinstance(transformer, dict)
            else None
        )

        # Add a new and possibly transformed `category`column, group by column and apply aggreates
        document_index = (
            self._document_index.assign(category=transform)
            .groupby('category')
            .agg(count_aggregates)
            .rename(columns={column_name: 'n_docs'})
        )

        # Reset column index to a single level
        document_index.columns = [col if isinstance(col, str) else '_'.join(col) for col in document_index.columns]

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
                    'category': index_values,
                    'filename': [f"{column_name}_{value}.txt" for value in index_values],
                    'document_name': [f"{column_name}_{value}" for value in index_values],
                }
            ).set_index('category'),
            document_index,
            how='left',
            left_index=True,
            right_index=True,
        )

        # Add `year` column if grouping column was 'year`, set to index or min year based on grouping column
        document_index['year'] = document_index.index if column_name == 'year' else document_index.year_min

        # Add `document_id`
        document_index = document_index.reset_index()
        document_index['document_id'] = document_index.index

        # Set `document_name` as index of result data frame
        document_index = document_index.set_index('document_name', drop=False).rename_axis('')

        return DocumentIndex(document_index)

    def set_strictly_increasing_index(self) -> "DocumentIndex":
        self._document_index['document_id'] = get_strictly_increasing_document_id(
            self._document_index, document_id_field=None
        )
        self._document_index = self._document_index.set_index('document_id', drop=False).rename_axis('')
        return self


def get_strictly_increasing_document_id(
    document_index: pd.DataFrame, document_id_field: str = 'document_id'
) -> pd.Series:
    """[summary]

    Args:
        document_index (pd.DataFrame): [description]
        document_id_field (str): [description]

    Returns:
        pd.Series: [description]
    """

    if document_id_field in document_index.columns:
        if is_strictly_increasing(document_index[document_id_field]):
            return document_index[document_id_field]

    if is_strictly_increasing(document_index.index):
        return document_index.index

    return document_index.reset_index().index


def store_document_index(document_index: pd.DataFrame, filename: str):
    """[summary]

    Args:
        document_index (pd.DataFrame): [description]
        filename (str): [description]
    """
    document_index.to_csv(filename, sep='\t', header=True)


def load_document_index(
    filename: Union[str, StringIO], *, sep: str, document_id_field: str = 'document_id'
) -> pd.DataFrame:
    """Loads a document index and sets `document_name` as index column. Also adds `document_id`"""

    if filename is None:
        return None

    if isinstance(filename, pd.DataFrame):
        document_index = filename
    else:
        document_index: pd.DataFrame = pd.read_csv(filename, sep=sep)

    for old_or_unnamed_index_column in ['Unnamed: 0', 'filename.1']:
        if old_or_unnamed_index_column in document_index.columns:
            document_index = document_index.drop(old_or_unnamed_index_column, axis=1)

    if 'filename' not in document_index.columns:
        raise DocumentIndexError("expected mandatory column `filename` in document index, found no such thing")

    document_index['document_id'] = get_strictly_increasing_document_id(document_index, document_id_field)

    if 'document_name' not in document_index.columns or (document_index.document_name == document_index.filename).all():
        document_index['document_name'] = document_index.filename.apply(strip_path_and_extension)

    document_index = document_index.set_index('document_name', drop=False).rename_axis('')

    return document_index


def metadata_to_document_index(metadata: List[Dict], *, document_id_field: str = 'document_id') -> pd.DataFrame:
    """Creates a document index from collected filename fields metadata."""

    if metadata is None or len(metadata) == 0:
        metadata = {'filename': [], 'document_id': []}

    document_index = load_document_index(pd.DataFrame(metadata), sep=None, document_id_field=document_id_field)

    return document_index


def load_document_index_from_str(data_str: str, sep: str, document_id_field: str = 'document_id') -> pd.DataFrame:
    df = load_document_index(StringIO(data_str), sep=sep, document_id_field=document_id_field)
    return df


def consolidate_document_index(document_index: pd.DataFrame, reader_index: pd.DataFrame):
    """Returns a consolidated document index from an existing index, if exists,
    and the reader index."""

    if document_index is not None:
        columns = [x for x in reader_index.columns if x not in document_index.columns]
        if len(columns) > 0:
            document_index = document_index.merge(reader_index[columns], left_index=True, right_index=True, how='left')
        return document_index

    return reader_index


def document_index_upgrade(document_index: pd.DataFrame) -> pd.DataFrame:
    """Fixes older versions of document indexes"""

    if 'document_name' not in document_index.columns:
        document_index['document_name'] = document_index.filename.apply(strip_path_and_extension)

    if document_index.index.dtype == np.dtype('int64'):

        if 'document_id' not in document_index.columns:
            document_index['document_id'] = document_index.index

    document_index = document_index.set_index('document_name', drop=False).rename_axis('')

    return document_index


def add_document_index_attributes(*, catalogue: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """ Adds document meta data to given data frame (must have a document_id) """
    df = target.merge(catalogue, how='inner', left_on='document_id', right_on='document_id')
    return df


def update_document_index_token_counts(
    document_index: pd.DataFrame, doc_token_counts: List[Tuple[str, int, int]]
) -> pd.DataFrame:
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


def update_document_index_properties(document_index, *, document_name: str, property_bag: Mapping[str, int]):
    """[summary]

    Args:
        document_index ([type]): [description]
        document_name (str): [description]
        property_bag (Mapping[str, int]): [description]
    """
    property_bag = {k: property_bag[k] for k in property_bag if k not in ['document_name']}
    for key in [k for k in property_bag if k not in document_index.columns]:
        document_index.insert(len(document_index.columns), key, np.nan)
    document_index.update(pd.DataFrame(data=property_bag, index=[document_name], dtype=np.int64))
