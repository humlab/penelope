import os

from loguru import logger
from penelope.co_occurrence import ContextOpts, store_co_occurrences
from penelope.co_occurrence.interface import ComputeResult
from penelope.co_occurrence.partition_by_document import compute_corpus_co_occurrence
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, readers
from penelope.corpus.utils import generate_token2id

jj = os.path.join


concept = {}
n_context_width = 2
tokens_reader = readers.SparvCsvTokenizer(
    '/data/westac/data/riksdagens-protokoll.1920-2019.sparv4.csv.zip',
    extract_tokens_opts=ExtractTaggedTokensOpts(
        pos_includes='NN|PM|VB',
        pos_excludes='MAD|MID|PAD',
        pos_paddings="AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO",
        lemmatize=True,
        to_lowercase=True,
    ),
    reader_opts=TextReaderOpts(
        filename_fields="year:_:1",
    ),
)
token2id = generate_token2id(tokens for _, tokens in tokens_reader)
compute_result: ComputeResult = compute_corpus_co_occurrence(
    stream=tokens_reader,
    document_index=tokens_reader.document_index,
    token2id=token2id,
    context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=n_context_width),
    global_threshold_count=None,
    ignore_pad=None,
)

logger.info("Storing data")

store_co_occurrences("tests/output/riksdagens-protokoll.1920-2019.co-occurrences.zip", compute_result.co_occurrences)
compute_result.token2id.store("tests/output/riksdagens-protokoll.1920-2019.dictionary.zip")
