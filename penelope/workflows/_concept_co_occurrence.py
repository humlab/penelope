import json
import os
from typing import Any, Dict, Sequence

import pandas as pd
from penelope.co_occurrence import (
    ContextOpts,
    partitioned_corpus_co_occurrence,
    store_co_occurrences,
    to_vectorized_corpus,
)
from penelope.corpus import SparvTokenizedCsvCorpus, TokensTransformOpts, VectorizedCorpus
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts
from penelope.utility import replace_extension, strip_path_and_extension

from .utils import WorkflowException

# pylint: disable=too-many-arguments


def execute_workflow(
    input_filename: str,
    output_filename: str,
    *,
    context_opts: ContextOpts = None,
    extract_tokens_opts: ExtractTaggedTokensOpts = None,
    tokens_transform_opts: TokensTransformOpts = None,
    count_threshold: int = None,
    partition_keys: Sequence[str],
    filename_field: Any = None,
    store_vectorized: bool = False,
) -> pd.DataFrame:
    """Creates concept co-occurrence using specified options and stores a co-occurrence CSV file
    and optionally a vectorized corpus.

    Parameters
    ----------
    input_filename : str
        Sparv v4 input corpus in CSV export format
    output_filename : str
        Target co-occurrence CSV file, optionally compressed if extension is ".zip"
    partition_keys : Sequence[str]
        Key in corpus document index to use to split corpus in sub-corpora.
        Each sub-corpus co-occurrence is associated to the corresponding key value.
        Usually the `year` column in the document index.
    context_opts: ContextOpts
        context_width : int, optional
            Width of context i.e. distance to cencept word, by default None
        concept : List[str], optional
            Tokens that defines the concept, by default None
        no_concept : bool, optional
            Specifies if concept should be removed from result, by default False
    count_threshold : int, optional
        Word pair count threshold (entire corpus, by default None
    extract_tokens_opts : ExtractTaggedTokensOpts, optional
    tokens_transform_opts : TokensTransformOpts, optional
    filename_field : Any, optional
        Specifies fields to extract from document's filename, by default None
    store_vectorized : bool, optional
        If true, then the co-occurrence pairs are stored in a vectorized corpus
        with a vocabulary consisting of "word1/word2" tokens, by default False

    Raises
    ------
    WorkflowException
        When any argument check fails.
    """
    if len(context_opts.concept or []) == 0:
        raise WorkflowException("please specify at least one concept (--concept e.g. --concept=information)")

    if len(filename_field or []) == 0:
        raise WorkflowException(
            "please specify at least one filename field (--filename-field e.g. --filename-field='year:_:1')"
        )

    if context_opts.context_width is None:
        raise WorkflowException(
            "please specify at width of context as max distance from cencept (--context-width e.g. --context_width=2)"
        )

    if len(partition_keys or []) == 0:
        raise WorkflowException("please specify partition key) (--partition-key e.g --partition-key=year)")

    if len(partition_keys) > 1:
        raise WorkflowException("only one partition key is allowed (for now)")

    reader_opts = TextReaderOpts(
        filename_pattern='*.csv',
        filename_fields=filename_field,  # use filename
        index_field=None,
        as_binary=False,
    )

    corpus: SparvTokenizedCsvCorpus = SparvTokenizedCsvCorpus(
        source=input_filename,
        reader_opts=reader_opts,
        extract_tokens_opts=extract_tokens_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    token2id = corpus.token2id  # make one pass to create vocabulary and gather token counts
    document_index = corpus.document_index

    co_occurrences = partitioned_corpus_co_occurrence(
        stream=corpus,
        token2id=token2id,
        document_index=document_index,
        context_opts=context_opts,
        global_threshold_count=count_threshold,
        partition_column=partition_keys[0],
    )

    store_concept_co_occurrence_bundle(
        output_filename,
        store_vectorized=store_vectorized,
        input_filename=input_filename,
        partition_keys=partition_keys,
        count_threshold=count_threshold,
        co_occurrences=co_occurrences,
        reader_opts=reader_opts,
        tokens_transform_opts=tokens_transform_opts,
        context_opts=context_opts,
        extract_tokens_opts=extract_tokens_opts,
    )

    return co_occurrences


def store_concept_co_occurrence_bundle(
    output_filename: str,
    *,
    store_vectorized: bool,
    input_filename: str,
    partition_keys: Sequence[str],
    count_threshold: int = None,
    co_occurrences: pd.DataFrame,
    reader_opts: Dict,
    tokens_transform_opts: TokensTransformOpts,
    context_opts: ContextOpts,
    extract_tokens_opts: ExtractTaggedTokensOpts,
):

    store_co_occurrences(output_filename, co_occurrences)

    if store_vectorized:
        v_corpus: VectorizedCorpus = to_vectorized_corpus(co_occurrences=co_occurrences, value_column='value_n_t')
        v_corpus.dump(tag=strip_path_and_extension(output_filename), folder=os.path.split(output_filename)[0])

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
