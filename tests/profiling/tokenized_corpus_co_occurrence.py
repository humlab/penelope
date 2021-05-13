import os

from penelope.co_occurrence import ContextOpts
from penelope.co_occurrence.interface import ComputeResult
from penelope.co_occurrence.partition_by_document import partition_by_document_co_occurrence
from penelope.corpus import ExtractTaggedTokensOpts, SparvTokenizedCsvCorpus, TextReaderOpts

jj = os.path.join


concept = {}
n_context_width = 2
corpus = SparvTokenizedCsvCorpus(
    '../test_data/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip',
    reader_opts=TextReaderOpts(
        filename_fields="year:_:1",
    ),
    extract_tokens_opts=ExtractTaggedTokensOpts(
        pos_includes='NN|PM|VB',
        pos_excludes='MAD|MID|PAD',
        pos_paddings="AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO",
        lemmatize=True,
    ),
)
compute_result: ComputeResult = partition_by_document_co_occurrence(
    stream=corpus,
    document_index=corpus.document_index,
    token2id=corpus.token2id,
    context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=n_context_width),
    # transform_opts=None,
    global_threshold_count=None,
    # partition_key='document_name',
    ignore_pad=None,
)

assert compute_result is not None
assert len(compute_result.co_occurrences) > 0
