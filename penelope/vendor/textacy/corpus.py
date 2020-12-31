import os
import types
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import penelope.corpus.readers.text_tokenizer as text_tokenizer
import textacy
from penelope.corpus import TextTransformOpts
from penelope.corpus.readers.interfaces import ICorpusReader, TextReaderOpts
from penelope.utility import (
    FilenameFieldSpecs,
    extract_filename_metadata,
    getLogger,
    lists_of_dicts_merged_by_key,
    noop,
    path_add_suffix,
    timecall,
)
from spacy.language import Language as SpacyLanguage
from spacy.tokens import Doc as SpacyDoc
from textacy import Corpus as TextacyCorpus

from .language import create_nlp

logger = getLogger('corpus_text_analysis')

# pylint: disable=too-many-arguments


def create_corpus(
    reader: ICorpusReader,
    nlp: SpacyLanguage,
    *,
    extra_metadata: List[Dict[str, Any]] = None,
    tick: Callable = noop,
    n_chunk_threshold: int = 100000,
) -> TextacyCorpus:

    corpus: TextacyCorpus = textacy.Corpus(nlp)
    counter = 0

    metadata_mapping = {
        x['filename']: x for x in lists_of_dicts_merged_by_key(reader.metadata, extra_metadata, key='filename')
    }

    for filename, text in reader:

        metadata = metadata_mapping[filename]

        if len(text) > n_chunk_threshold:
            doc: SpacyDoc = textacy.spacier.utils.make_doc_from_text_chunks(
                text, lang=nlp, chunk_size=n_chunk_threshold
            )
            corpus.add_doc(doc)
            doc._.meta = metadata
        else:
            corpus.add((text, metadata))

        counter += 1
        if counter % 100 == 0:
            logger.info('%s documents added...', counter)

        tick(counter)

    return corpus


@timecall
def save_corpus(
    corpus: textacy.Corpus, filename: str, lang=None, include_tensor: bool = False
):  # pylint: disable=unused-argument
    if not include_tensor:
        for doc in corpus:
            doc.tensor = None
    corpus.save(filename)


@timecall
def load_corpus(filename: str, lang: str) -> textacy.Corpus:  # pylint: disable=unused-argument
    corpus = textacy.Corpus.load(lang, filename)
    return corpus


def merge_named_entities(corpus: textacy.Corpus):
    logger.info('Working: Merging named entities...')
    try:
        for doc in corpus:
            named_entities = textacy.extract.entities(doc)
            textacy.spacier.utils.merge_spans(named_entities, doc)
    except TypeError as ex:
        logger.error(ex)
        logger.info('NER merge failed')


def generate_corpus_filename(
    source_path: str,
    language: str,
    nlp_args=None,
    preprocess_args=None,
    compression: str = 'bz2',
    extension: str = 'bin',
) -> str:
    nlp_args = nlp_args or {}
    preprocess_args = preprocess_args or {}
    disabled_pipes = nlp_args.get('disable', ())
    suffix = '_{}_{}{}'.format(
        language,
        '_'.join([k for k in preprocess_args if preprocess_args[k]]),
        '_disable({})'.format(','.join(disabled_pipes)) if len(disabled_pipes) > 0 else '',
    )
    filename = path_add_suffix(source_path, suffix, new_extension='.' + extension)
    if (compression or '') != '':
        filename += '.' + compression
    return filename


def _get_document_metadata(
    filename: str,
    metadata: Dict[str, Any] = None,
    document_index: pd.DataFrame = None,
    document_columns: List[str] = None,
    filename_fields: FilenameFieldSpecs = None,
) -> Mapping[str, Any]:
    """Extract document metadata from filename and document index"""
    metadata = metadata or {}

    if filename_fields is not None:

        metadata = {**metadata, **extract_filename_metadata(filename=filename, filename_fields=filename_fields)}

    if document_index is not None:

        if 'filename' not in document_index.columns:
            raise ValueError("Filename field 'filename' not found in document index")

        document_row = document_index[document_index.filename == filename]

        if document_columns is not None:
            document_row = document_row[document_columns]

        if len(document_row) == 0:
            raise ValueError(f"Name '{filename}' not found in index")

        metadata = {**metadata, **(document_row.iloc[0].to_dict())}

        if 'document_id' not in metadata:
            metadata['document_id'] = document_row.index[0]

    return metadata


def _extend_stream_with_metadata(
    tokens_reader: text_tokenizer.TextTokenizer,
    document_index: pd.DataFrame = None,
    document_columns: List[str] = None,
    filename_fields: FilenameFieldSpecs = None,
) -> Iterable[Tuple[str, str, Dict]]:
    """Extract and adds document meta data to stream

    Parameters
    ----------
    tokens_reader : text_tokenizer.TextTokenizer
        Reader, returns stream of filename and tokens
    document_index : pd.DataFrame, optional
        Document index, by default None
    document_columns : List[str], optional
        Columns in document index, by default None
    filename_fields : FilenameFieldSpecs, optional
        Filename fields to extract, by default None

    Yields
    -------
    Iterable[Tuple[str, str, Dict]]
        Stream augumented with meta data.
    """
    metadata_mapping = {x['filename']: x for x in tokens_reader.metadata}
    for filename, tokens in tokens_reader:

        metadata = _get_document_metadata(
            filename,
            metadata=metadata_mapping[filename],
            document_index=document_index,
            document_columns=document_columns,
            filename_fields=filename_fields,
        )

        yield filename, ' '.join(tokens), metadata


def load_or_create(
    source_path: Any,
    language: str,
    *,
    document_index: pd.DataFrame = None,  # data_frame or lambda corpus: corpus_index
    merge_entities: bool = False,
    overwrite: bool = False,
    binary_format: bool = True,
    use_compression: bool = True,
    disabled_pipes: List[str] = None,
    filename_fields: FilenameFieldSpecs = None,
    document_columns: List[str] = None,
    tick=noop,
) -> Dict[str, Any]:
    """Loads textaCy corpus from disk if it exists on disk with a name that satisfies the given arguments.
    Otherwise creates a new corpus and adds metadata to corpus document index as specified by `filename_fields` and/or document index.

    Parameters
    ----------
    source_path : Any
        Corpus path name.
    language : str
        The spaCy language designator.
    document_index : pd.DataFrame, optional
        A document index (if specified, then must include a `filename` column), by default None
    overwrite : bool, optional
        Force recreate of corpus if it exists on disk, by default False
    binary_format : bool, optional
        Store in pickled binary format, by default True
    use_compression : bool, optional
        Use compression, by default True
    disabled_pipes : List[str], optional
        SpaCy pipes that should be disabled, by default None
    filename_fields : FilenameFieldSpecs, optional
        Specifies metadata that should be extracted from filename, by default None
    document_columns : List[str], optional
        Columns in `document_index` to add to metadata, all columns will be added if None, by default None
    tick : Callable, optional
        Progress callback function, by default noop

    Returns
    -------
    Dict[str,Any]
        source_path         Source corpus path
        language            spaCy language specifier
        nlp                 spaCy nlp instance
        textacy_corpus      textaCy corpus
        textacy_corpus_path textaCy corpus filename
    """
    tick = tick or noop

    nlp_args = {'disable': disabled_pipes or []}

    textacy_corpus_path = generate_corpus_filename(
        source_path,
        language,
        nlp_args=nlp_args,
        extension='bin' if binary_format else 'pkl',
        compression='bz2' if use_compression else '',
    )

    nlp = create_nlp(language, **nlp_args)

    if overwrite or not os.path.isfile(textacy_corpus_path):

        logger.info('Computing new corpus %s...', textacy_corpus_path)

        tokens_streams = text_tokenizer.TextTokenizer(
            source=source_path,
            transform_opts=TextTransformOpts(fix_unicode=True, fix_accents=True),
            reader_opts=TextReaderOpts(filename_fields=filename_fields),
        )

        reader = _extend_stream_with_metadata(
            tokens_streams,
            document_index=document_index,
            document_columns=document_columns,
            filename_fields=None,  # n.b. fields extracted abve
        )

        logger.info('Stream created...')

        tick(0, len(tokens_streams.filenames))

        logger.info('Creating corpus (this might take some time)...')
        textacy_corpus = create_corpus(reader, nlp, tick=tick)

        logger.info('Storing corpus (this might take some time)...')
        save_corpus(textacy_corpus, textacy_corpus_path)

        tick(0)

    else:
        tick(1, 2)
        logger.info('...reading corpus (this might take several minutes)...')
        textacy_corpus = load_corpus(textacy_corpus_path, nlp)

    if merge_entities:
        merge_named_entities(textacy_corpus)

    tick(0)
    logger.info('Done!')

    return types.SimpleNamespace(
        source_path=source_path,
        language=language,
        nlp=nlp,
        corpus=textacy_corpus,
        corpus_path=textacy_corpus_path,
    )
