import json
import os
from dataclasses import dataclass
from typing import Mapping, Tuple, Union

import pandas as pd
import scipy
from penelope.corpus import CorpusVectorizer, ITokenizedCorpus, TokenizedCorpus, TokensTransformOpts, VectorizedCorpus
from penelope.corpus.document_index import DocumentIndex
from penelope.corpus.readers import ExtractTaggedTokensOpts, ICorpusReader, TextReaderOpts
from penelope.notebook.word_trends import TrendsData
from penelope.utility import getLogger, read_json, replace_extension, right_chop, strip_path_and_extension

from .interface import ContextOpts, CoOccurrenceError

logger = getLogger()


CO_OCCURRENCE_FILENAME_POSTFIX = '_co-occurrence.csv.zip'
CO_OCCURRENCE_FILENAME_PATTERN = f'*{CO_OCCURRENCE_FILENAME_POSTFIX}'


def filename_to_folder_and_tag(co_occurrences_filename: str) -> Tuple[str, str]:
    """Strips out corpus folder and tag from filename having CO_OCCURRENCE_FILENAME_POSTFIX ending"""
    corpus_folder, corpus_basename = os.path.split(co_occurrences_filename)
    corpus_tag = right_chop(corpus_basename, CO_OCCURRENCE_FILENAME_POSTFIX)
    return corpus_folder, corpus_tag


def tag_to_filename(*, tag: str) -> str:
    return f"{tag}{CO_OCCURRENCE_FILENAME_POSTFIX}"


def folder_and_tag_to_filename(*, folder: str, tag: str) -> str:
    return os.path.join(folder, tag_to_filename(tag=tag))


def store_co_occurrences(filename: str, df: pd.DataFrame):
    """Store co-occurrence result data to CSV-file"""

    if filename.endswith('zip'):
        archive_name = f"{strip_path_and_extension(filename)}.csv"
        compression = dict(method='zip', archive_name=archive_name)
    else:
        compression = 'infer'

    df.to_csv(filename, sep='\t', header=True, compression=compression, decimal=',')

    # pandas_to_csv_zip(
    #     zip_filename=co_occurrence_filename,
    #     dfs=(co_occurrences, 'co_occurrence.csv'),
    #     extension='csv',
    #     header=True,
    #     sep="\t",
    #     decimal=',',
    #     quotechar='"',
    # )


def load_co_occurrences(filename: str) -> pd.DataFrame:
    """Load co-occurrences from CSV-file"""
    df = pd.read_csv(filename, sep='\t', header=0, decimal=',', index_col=0)

    return df


def to_vectorized_corpus(
    co_occurrences: pd.DataFrame, document_index: pd.DataFrame, value_column: str = "value"
) -> VectorizedCorpus:
    """Creates a DTM corpus from a co-occurrence result set that was partitioned by year."""
    # Create new tokens from the co-occurring pairs
    tokens = co_occurrences.apply(lambda x: f'{x["w1"]}/{x["w2"]}', axis=1)

    # Create a vocabulary & token2id mapping
    token2id = {w: i for i, w in enumerate(sorted([w for w in set(tokens)]))}

    # Create a year to index mapping (i.e. year to document_id)
    year2index = document_index.set_index('year').document_id.to_dict()

    df_yearly_weights = pd.DataFrame(
        data={
            'year_index': co_occurrences.year.apply(lambda y: year2index[y]),
            'token_id': tokens.apply(lambda x: token2id[x]),
            'weight': co_occurrences[value_column],
        }
    )
    # Make certain  matrix gets right shape (otherwise empty documents at the end reduces row count)
    shape = (len(year2index), len(token2id))
    coo_matrix = scipy.sparse.coo_matrix(
        (df_yearly_weights.weight, (df_yearly_weights.year_index, df_yearly_weights.token_id)),
        shape=shape,
    )

    document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

    v_corpus = VectorizedCorpus(coo_matrix, token2id=token2id, document_index=document_index)

    return v_corpus


def to_co_occurrence_matrix(
    corpus_or_reader: Union[ICorpusReader, TokenizedCorpus], vocabulary: Mapping[str, int] = None
) -> scipy.sparse.spmatrix:
    """Computes a term-term co-ocurrence matrix for documents in corpus/reader.

    Parameters
    ----------
    corpus_or_reader : Union[ICorpusReader,TokenizedCorpus]
        Sequence of tokenized documents

    Returns
    -------
    pd.DataFrame
        Upper diagonal of term-term frequency matrix (TTM). Note that diagonal (wi, wi) is not returned
    """

    if not isinstance(corpus_or_reader, ITokenizedCorpus):
        corpus_or_reader = TokenizedCorpus(reader=corpus_or_reader)

    vocabulary = vocabulary or corpus_or_reader.token2id
    dtm_corpus = CorpusVectorizer().fit_transform(corpus_or_reader, already_tokenized=True, vocabulary=vocabulary)
    term_term_matrix = dtm_corpus.co_occurrence_matrix()
    return term_term_matrix


def to_dataframe(
    term_term_matrix: scipy.sparse.spmatrix,
    id2token: Mapping[int, str],
    document_index: pd.DataFrame = None,
    threshold_count: int = 1,
) -> pd.DataFrame:
    """Converts a TTM to a Pandas DataFrame

    Parameters
    ----------
    term_term_matrix : scipy.sparse.spmatrix
        [description]
    id2token : Mapping[int,str]
        [description]
    document_index : pd.DataFrame, optional
        [description], by default None
    threshold_count : int, optional
        Min count (`value`) to include in result, by default 1

    Returns
    -------
    [type]
        [description]
    """

    if 'n_tokens' not in document_index.columns:
        raise CoOccurrenceError("expected `document_index.n_tokens`, but found no column")

    if 'n_raw_tokens' not in document_index.columns:
        raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no column")

    coo_df = (
        pd.DataFrame({'w1_id': term_term_matrix.row, 'w2_id': term_term_matrix.col, 'value': term_term_matrix.data})[
            ['w1_id', 'w2_id', 'value']
        ]
        .sort_values(['w1_id', 'w2_id'])
        .reset_index(drop=True)
    )

    if threshold_count > 0:
        coo_df = coo_df[coo_df.value >= threshold_count]

    if document_index is not None:

        coo_df['value_n_d'] = coo_df.value / float(len(document_index))

        for n_token_count, target_field in [('n_tokens', 'value_n_t'), ('n_raw_tokens', 'value_n_r_t')]:

            if n_token_count in document_index.columns:
                try:
                    coo_df[target_field] = coo_df.value / float(sum(document_index[n_token_count].values))
                except ZeroDivisionError:
                    coo_df[target_field] = 0.0
            else:
                logger.warning(f"{target_field}: cannot compute since {n_token_count} not in corpus document catalogue")

    coo_df['w1'] = coo_df.w1_id.apply(lambda x: id2token[x])
    coo_df['w2'] = coo_df.w2_id.apply(lambda x: id2token[x])

    coo_df = coo_df[['w1', 'w2', 'value', 'value_n_d', 'value_n_t']]

    return coo_df


@dataclass
class Bundle:

    co_occurrences_filename: str = None
    corpus_folder: str = None
    corpus_tag: str = None

    co_occurrences: pd.DataFrame = None
    document_index: pd.DataFrame = None
    corpus: VectorizedCorpus = None
    compute_options: dict = None


def store_bundle(output_filename: str, bundle: Bundle) -> Bundle:

    store_co_occurrences(output_filename, bundle.co_occurrences)

    if bundle.corpus is not None:
        if bundle.corpus_tag is None:
            bundle.corpus_tag = strip_path_and_extension(output_filename)
        bundle.corpus_folder = os.path.split(output_filename)[0]
        bundle.corpus.dump(tag=bundle.corpus_tag, folder=bundle.corpus_folder)
        DocumentIndex(bundle.document_index).store(
            os.path.join(bundle.corpus_folder, f"{bundle.corpus_tag}_document_index.csv")
        )

    with open(replace_extension(output_filename, 'json'), 'w') as json_file:
        json.dump(bundle.compute_options, json_file, indent=4)


def load_bundle(co_occurrences_filename: str, compute_corpus: bool = True) -> "Bundle":
    """Loads bundle identified by given filename i.e. `corpus_folder`/`corpus_tag`_co-occurrence.csv.zip"""

    corpus_folder, corpus_tag = filename_to_folder_and_tag(co_occurrences_filename)
    co_occurrences = load_co_occurrences(co_occurrences_filename)
    document_index = DocumentIndex.load(os.path.join(corpus_folder, f"{corpus_tag}_document_index.csv")).document_index
    corpus = (
        VectorizedCorpus.load(folder=corpus_folder, tag=corpus_tag)
        if VectorizedCorpus.dump_exists(folder=corpus_folder, tag=corpus_tag)
        else None
    )
    if corpus is None and compute_corpus:
        corpus = to_vectorized_corpus(
            co_occurrences=co_occurrences, document_index=document_index, value_column='value'
        )

    options = load_options(co_occurrences_filename)

    bundle = Bundle(
        co_occurrences_filename=co_occurrences_filename,
        corpus_folder=corpus_folder,
        corpus_tag=corpus_tag,
        co_occurrences=co_occurrences,
        document_index=document_index,
        compute_options=options,
        corpus=corpus,
    )

    return bundle


def to_trends_data(bundle: Bundle, n_count=25000):

    trends_data = TrendsData(
        compute_options=bundle.compute_options,
        corpus=bundle.corpus,
        corpus_folder=bundle.corpus_folder,
        corpus_tag=bundle.corpus_tag,
        n_count=n_count,
    ).remember(co_occurrences=bundle.co_occurrences, document_index=bundle.document_index)

    return trends_data


def load_options(co_occurrences_filename: str) -> dict:
    """Loads options used when co-occurrence was computed"""

    options_filename = replace_extension(co_occurrences_filename, 'json')
    if os.path.isfile(options_filename):
        return read_json(options_filename)
    return {'not_found': options_filename}


def create_options_bundle(
    *,
    reader_opts: TextReaderOpts,
    tokens_transform_opts: TokensTransformOpts,
    context_opts: ContextOpts,
    extract_tokens_opts: ExtractTaggedTokensOpts,
    **other_options,
):
    options = {
        **{
            'reader_opts': reader_opts.props,
            'context_opts': context_opts.props,
            'tokens_transform_opts': tokens_transform_opts.props,
            'extract_tokens_opts': extract_tokens_opts.props,
        },
        **other_options,
    }
    return options
