import json
import os
from typing import Any, List, Tuple

import penelope.co_occurrence.concept_co_occurrence as concept_co_occurrence
from penelope.corpus.readers.annotation_opts import AnnotationOpts
from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus
from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.utility import replace_extension, strip_path_and_extension

from .utils import WorkflowException

# pylint: disable=too-many-arguments


def execute_workflow(
    input_filename: str,
    output_filename: str,
    *,
    concept: List[str] = None,
    no_concept: bool = False,
    count_threshold: int = None,
    context_width: int = None,
    partition_keys: Tuple[str, List[str]],
    annotation_opts: AnnotationOpts = None,
    filename_field: Any = None,
    store_vectorized: bool = False,
    tokens_transform_opts: TokensTransformOpts = None,
):
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
    concept : List[str], optional
        Tokens that defines the concept, by default None
    no_concept : bool, optional
        Specifies if concept should be removed from result, by default False
    count_threshold : int, optional
        Word pair count threshold (entire corpus, by default None
    context_width : int, optional
        Width of context i.e. distance to cencept word, by default None
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
    if len(concept or []) == 0:
        raise WorkflowException("please specify at least one concept (--concept e.g. --concept=information)")

    if len(filename_field or []) == 0:
        raise WorkflowException(
            "please specify at least one filename field (--filename-field e.g. --filename-field='year:_:1')"
        )

    if context_width is None:
        raise WorkflowException(
            "please specify at width of context as max distance from cencept (--context-width e.g. --context_width=2)"
        )

    if len(partition_keys or []) == 0:
        raise WorkflowException("please specify partition key(s) (--partition-key e.g --partition-key=year)")

    tokenizer_opts = {'filename_pattern': '*.csv', 'filename_fields': filename_field, 'as_binary': False}

    corpus = SparvTokenizedCsvCorpus(
        source=input_filename,
        tokenizer_opts=tokenizer_opts,
        annotation_opts=annotation_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    coo_df = concept_co_occurrence.partitioned_corpus_concept_co_occurrence(
        corpus,
        concepts=concept,
        no_concept=no_concept,
        n_count_threshold=count_threshold,
        n_context_width=context_width,
        partition_keys=partition_keys,
    )

    concept_co_occurrence.store_co_occurrences(output_filename, coo_df)

    if store_vectorized:
        v_corpus = concept_co_occurrence.to_vectorized_corpus(co_occurrences=coo_df, value_column='value_n_t')
        v_corpus.dump(tag=strip_path_and_extension(output_filename), folder=os.path.split(output_filename)[0])

    with open(replace_extension(output_filename, 'json'), 'w') as json_file:
        store_options = {
            'input': input_filename,
            'output': output_filename,
            'concepts': list(concept),
            'no_concept': no_concept,
            'n_count_threshold': count_threshold,
            'n_context_width': context_width,
            'partition_keys': partition_keys,
            'tokenizer_opts': tokenizer_opts,
            'tokens_transform_opts': tokens_transform_opts.props,
            'annotation_opts': annotation_opts.props,
        }
        json.dump(store_options, json_file, indent=4)
