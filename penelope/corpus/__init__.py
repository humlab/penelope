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
from .purgatory.segmented_text_corpus import ChunkSegmenter, DocumentSegmenter, SegmentedTextCorpus, SentenceSegmenter
from .purgatory.text_lines_corpus import SimpleTextLinesCorpus
from .readers import (
    ExtractTaggedTokensOpts,
    FilenameFilterSpec,
    ICorpusReader,
    PandasCorpusReader,
    PhraseSubstitutions,
    TextReader,
    TextReaderOpts,
    TextSource,
    TokenizeTextReader,
    ZipCorpusReader,
    streamify_text_source,
)
from .sparv.sparv_corpus import (
    SparvTokenizedCsvCorpus,
    SparvTokenizedXmlCorpus,
    sparv_csv_extract_and_store,
    sparv_xml_extract_and_store,
)
from .token2id import ClosedVocabularyError, Token2Id, id2token2token2id
from .tokenized_corpus import ReiterableTerms, TokenizedCorpus
from .transform import (
    TextTransformer,
    TextTransformOpts,
    TokensTransformOpts,
    Transform,
    TransformRegistry,
    default_tokenizer,
    dehyphen,
    tokens_transformer,
)
from .utils import bow2text, csr2bow, generate_token2id, term_frequency
