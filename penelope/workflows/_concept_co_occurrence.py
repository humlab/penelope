import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from penelope.co_occurrence import (
    ConceptContextOpts,
    partitioned_corpus_concept_co_occurrence,
    store_co_occurrences,
    to_vectorized_corpus,
)
from penelope.corpus.readers import AnnotationOpts
from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus
from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.utility import replace_extension, strip_path_and_extension

from .utils import WorkflowException

# pylint: disable=too-many-arguments


def execute_workflow(
    input_filename: str,
    output_filename: str,
    *,
    concept_opts: ConceptContextOpts = None,
    annotation_opts: AnnotationOpts = None,
    tokens_transform_opts: TokensTransformOpts = None,
    count_threshold: int = None,
    partition_keys: Tuple[str, List[str]],
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
    partition_keys : Tuple[str, List[str]]
        Key in corpus document index to use to split corpus in sub-corpora.
        Each sub-corpus co-occurrence is associated to corresponding key value.
        Usually the `year` column in the document index.
    concept_opts: ConceptContextOpts
        concept : List[str], optional
            Tokens that defines the concept, by default None
        no_concept : bool, optional
            Specifies if concept should be removed from result, by default False
        context_width : int, optional
            Width of context i.e. distance to cencept word, by default None
    count_threshold : int, optional
        Word pair count threshold (entire corpus, by default None
    annotation_opts : AnnotationOpts, optional
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
    if len(concept_opts.concept or []) == 0:
        raise WorkflowException("please specify at least one concept (--concept e.g. --concept=information)")

    if len(filename_field or []) == 0:
        raise WorkflowException(
            "please specify at least one filename field (--filename-field e.g. --filename-field='year:_:1')"
        )

    if concept_opts.context_width is None:
        raise WorkflowException(
            "please specify at width of context as max distance from cencept (--context-width e.g. --context_width=2)"
        )

    if len(partition_keys or []) == 0:
        raise WorkflowException("please specify partition key(s) (--partition-key e.g --partition-key=year)")

    tokenizer_opts = {
        'filename_pattern': '*.csv',
        'filename_fields': filename_field,
        'as_binary': False,
    }

    corpus = SparvTokenizedCsvCorpus(
        source=input_filename,
        tokenizer_opts=tokenizer_opts,
        annotation_opts=annotation_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    co_occurrences = partitioned_corpus_concept_co_occurrence(
        corpus,
        concept_opts=concept_opts,
        global_threshold_count=count_threshold,
        partition_keys=partition_keys,
    )

    store_concept_co_occurrence_bundle(
        output_filename,
        store_vectorized=store_vectorized,
        input_filename=input_filename,
        partition_keys=partition_keys,
        count_threshold=count_threshold,
        co_occurrences=co_occurrences,
        tokenizer_opts=tokenizer_opts,
        tokens_transform_opts=tokens_transform_opts,
        concept_opts=concept_opts,
        annotation_opts=annotation_opts,
    )

    return co_occurrences


def store_concept_co_occurrence_bundle(
    output_filename: str,
    *,
    store_vectorized: bool,
    input_filename: str,
    partition_keys: Tuple[str, List[str]],
    count_threshold: int = None,
    co_occurrences: pd.DataFrame,
    tokenizer_opts: Dict,
    tokens_transform_opts: TokensTransformOpts,
    concept_opts: ConceptContextOpts,
    annotation_opts: AnnotationOpts,
):

    store_co_occurrences(output_filename, co_occurrences)

    if store_vectorized:
        v_corpus = to_vectorized_corpus(co_occurrences=co_occurrences, value_column='value_n_t')
        v_corpus.dump(tag=strip_path_and_extension(output_filename), folder=os.path.split(output_filename)[0])

    with open(replace_extension(output_filename, 'json'), 'w') as json_file:
        store_options = {
            'input_filename': input_filename,
            'output_filename': output_filename,
            'partition_keys': partition_keys,
            'count_threshold': count_threshold,
            'tokenizer_opts': tokenizer_opts,
            'concept_opts': concept_opts.props,
            'tokens_transform_opts': tokens_transform_opts.props,
            'annotation_opts': annotation_opts.props,
        }

        json.dump(store_options, json_file, indent=4)
