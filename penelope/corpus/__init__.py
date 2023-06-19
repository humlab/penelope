# type: ignore
from .document_index import (
    DOCUMENT_INDEX_COUNT_COLUMNS,
    DocumentIndex,
    DocumentIndexHelper,
    consolidate_document_index,
    count_documents_in_index_by_pivot,
    document_index_upgrade,
    get_document_id,
    load_document_index,
    load_document_index_from_str,
    metadata_to_document_index,
    overload_by_document_index_properties,
    store_document_index,
    update_document_index_by_dicts_or_tuples,
    update_document_index_properties,
    update_document_index_token_counts,
    update_document_index_token_counts_by_corpus,
)
from .dtm import (
    CorpusVectorizer,
    IVectorizedCorpus,
    VectorizedCorpus,
    VectorizeOpts,
    find_matching_words_in_vocabulary,
    load_corpus,
    load_metadata,
    store_metadata,
)
from .interfaces import ICorpus, ITokenizedCorpus
from .readers.interfaces import ExtractTaggedTokensOpts, FilenameFilterSpec, PhraseSubstitutions, TextReaderOpts
from .segmented_text_corpus import ChunkSegmenter, DocumentSegmenter, SegmentedTextCorpus, SentenceSegmenter
from .sparv_corpus import (
    SparvTokenizedCsvCorpus,
    SparvTokenizedXmlCorpus,
    sparv_csv_extract_and_store,
    sparv_xml_extract_and_store,
)
from .text_lines_corpus import SimpleTextLinesCorpus
from .token2id import ClosedVocabularyError, Token2Id, id2token2token2id
from .tokenized_corpus import ReiterableTerms, TokenizedCorpus
from .tokens_transformer import TokensTransformOpts
from .transformer import TextTransformer, TextTransformOpts
from .transforms import (
    Transform,
    TransformRegistry,
    default_tokenizer,
    dehyphen,
    has_alphabetic,
    max_chars_factory,
    min_chars_factory,
    normalize_whitespace,
    only_alphabetic,
    only_any_alphanumeric,
    remove_empty,
    remove_numerals,
    remove_symbols,
    to_lower,
    to_upper,
)
from .utils import bow2text, csr2bow, generate_token2id, term_frequency
