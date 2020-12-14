import json
import os
from typing import Mapping, Sequence, Union

import pandas as pd
import scipy
from penelope.corpus import CorpusVectorizer, ITokenizedCorpus, TokenizedCorpus, TokensTransformOpts, VectorizedCorpus
from penelope.corpus.readers import ExtractTaggedTokensOpts, ICorpusReader, TextReaderOpts
from penelope.utility import getLogger, replace_extension, strip_path_and_extension

from .interface import ContextOpts

logger = getLogger()


def store_co_occurrences(filename: str, df: pd.DataFrame):
    """Store co-occurrence result data to CSV-file"""

    if filename.endswith('zip'):
        archive_name = f"{strip_path_and_extension(filename)}.csv"
        compression = dict(method='zip', archive_name=archive_name)
    else:
        compression = 'infer'

    df.to_csv(filename, sep='\t', header=True, compression=compression, decimal=',')


def load_co_occurrences(filename: str) -> pd.DataFrame:
    """Load co-occurrences from CSV-file"""
    df = pd.read_csv(filename, sep='\t', header=0, decimal=',', index_col=0)

    return df


def to_vectorized_corpus(co_occurrences: pd.DataFrame, value_column: str) -> VectorizedCorpus:

    # Create new tokens from the co-occurring pairs
    tokens = co_occurrences.apply(lambda x: f'{x["w1"]}/{x["w2"]}', axis=1)

    # Create a vocabulary
    vocabulary = list(sorted([w for w in set(tokens)]))

    # Create token2id mapping
    token2id = {w: i for i, w in enumerate(vocabulary)}
    years = list(sorted(co_occurrences.year.unique()))
    year2index = {year: i for i, year in enumerate(years)}

    df_yearly_weights = pd.DataFrame(
        data={
            'year_index': co_occurrences.year.apply(lambda y: year2index[y]),
            'token_id': tokens.apply(lambda x: token2id[x]),
            'weight': co_occurrences[value_column],
        }
    )

    coo_matrix = scipy.sparse.coo_matrix(
        (df_yearly_weights.weight, (df_yearly_weights.year_index, df_yearly_weights.token_id))
    )

    document_index = (
        pd.DataFrame(
            data={
                'document_id': list(range(0, len(years))),
                'filename': [f'{y}.coo' for y in years],
                'document_name': [f'{y}' for y in years],
                'year': years,
            }
        )
        .set_index('document_id', drop=False)
        .rename_axis('')
    )

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
    vectorizer = CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus_or_reader, already_tokenized=True, vocabulary=vocabulary)
    term_term_matrix = v_corpus.co_occurrence_matrix()

    return term_term_matrix


def to_dataframe(
    term_term_matrix: scipy.sparse.spmatrix,
    id2token: Mapping[int, str],
    catalogue: pd.DataFrame = None,
    threshold_count: int = 1,
):
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
    coo_df = (
        pd.DataFrame({'w1_id': term_term_matrix.row, 'w2_id': term_term_matrix.col, 'value': term_term_matrix.data})[
            ['w1_id', 'w2_id', 'value']
        ]
        .sort_values(['w1_id', 'w2_id'])
        .reset_index(drop=True)
    )

    if threshold_count > 0:
        coo_df = coo_df[coo_df.value >= threshold_count]

    if catalogue is not None:

        coo_df['value_n_d'] = coo_df.value / float(len(catalogue))

        for n_token_count, target_field in [('n_tokens', 'value_n_t'), ('n_raw_tokens', 'value_n_r_t')]:

            if n_token_count in catalogue.columns:
                try:
                    coo_df[target_field] = coo_df.value / float(sum(catalogue[n_token_count].values))
                except ZeroDivisionError:
                    coo_df[target_field] = 0.0
            else:
                logger.warning(f"{target_field}: cannot compute since {n_token_count} not in corpus document catalogue")

    coo_df['w1'] = coo_df.w1_id.apply(lambda x: id2token[x])
    coo_df['w2'] = coo_df.w2_id.apply(lambda x: id2token[x])

    coo_df = coo_df[['w1', 'w2', 'value', 'value_n_d', 'value_n_t']]

    return coo_df


def store_bundle(
    output_filename: str,
    co_occurrences: pd.DataFrame,
    corpus: VectorizedCorpus,
    corpus_tag: str,
    *,
    input_filename: str,
    partition_keys: Sequence[str],
    count_threshold: int,
    reader_opts: TextReaderOpts,
    tokens_transform_opts: TokensTransformOpts,
    context_opts: ContextOpts,
    extract_tokens_opts: ExtractTaggedTokensOpts,
):

    store_co_occurrences(output_filename, co_occurrences)

    if corpus is not None:
        if corpus_tag is None:
            corpus_tag = strip_path_and_extension(output_filename)
        corpus_folder = os.path.split(output_filename)[0]
        corpus.dump(tag=corpus_tag, folder=corpus_folder)

    with open(replace_extension(output_filename, 'json'), 'w') as json_file:
        store_options = {
            'input_filename': input_filename,
            'output_filename': output_filename,
            'partition_keys': partition_keys,
            'count_threshold': count_threshold,
            'reader_opts': reader_opts.props,
            'context_opts': context_opts.props,
            'tokens_transform_opts': tokens_transform_opts.props,
            'extract_tokens_opts': extract_tokens_opts.props,
        }
        json.dump(store_options, json_file, indent=4)
