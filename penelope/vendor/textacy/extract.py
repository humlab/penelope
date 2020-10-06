import penelope.utility as utility

from .utils import frequent_document_words, infrequent_words

logger = utility.getLogger('corpus_text_analysis')


def extract_document_terms(doc, extract_args):
    """Extracts documents and terms from a corpus

    Parameters
    ----------
    corpus : textacy Corpus
        Corpus in textacy format.

    extract_args : dict
        Dict that contains args that specifies the filter and transforms
        extract_args['args'] positional arguments for textacy.Doc.to_terms_list
        extract_args['kwargs'] Keyword arguments for textacy.Doc.to_terms_list
        extract_args['substitutions'] Dict (map) with term substitution
        extract_args['extra_stop_words'] List of additional stopwords to use

    Returns
    -------
    iterable of documents (which is iterable of terms)
        Documents where terms have ben filtered and transformed according to args.

    """
    kwargs = extract_args.get('kwargs', {})
    args = extract_args.get('args', {})

    extra_stop_words = set(extract_args.get('extra_stop_words', None) or [])
    substitutions = extract_args.get('substitutions', None)
    min_length = extract_args.get('min_length', 2)

    ngrams = args.get('ngrams', None)
    named_entities = args.get('named_entities', False)
    normalize = args.get('normalize', 'lemma')
    as_strings = args.get('as_strings', True)

    def tranform_token(w, substitutions=None):
        if '\n' in w:
            w = w.replace('\n', '_')
        if substitutions is not None and w in substitutions:
            w = substitutions[w]
        return w

    terms = (
        z
        for z in (
            tranform_token(w, substitutions)
            for w in doc._.to_terms_list(
                ngrams=ngrams, entities=named_entities, normalize=normalize, as_strings=as_strings, **kwargs
            )
            if len(w) >= min_length  # and w not in extra_stop_words
        )
        if z not in extra_stop_words
    )

    return terms


def extract_corpus_terms(corpus, extract_args):
    """Extracts documents and terms from a corpus

    Parameters
    ----------
    corpus : textacy Corpus
        Corpus in textacy format.

    extract_args : dict
        Dict that contains args that specifies the filter and transforms
        extract_args['args'] positional arguments for textacy.Doc.to_terms_list
        extract_args['kwargs'] Keyword arguments for textacy.Doc.to_terms_list
        extract_args['extra_stop_words'] List of additional stopwords to use
        extract_args['substitutions'] Dict (map) with term substitution
        DEPRECATED extract_args['mask_gpe'] Boolean flag indicating if GPE should be substituted
        extract_args['min_freq'] Integer value specifying min global word count.
        extract_args['max_doc_freq'] Float value between 0 and 1 indicating threshold
          for documentword frequency, Words that occur in more than `max_doc_freq`
          documents will be filtered out.

    None
    ----
        extract_args.min_freq and extract_args.min_freq is the same value but used differently
        kwargs.min_freq is passed directly as args to `textacy_doc.to_terms_list`
        tokens below extract_args.min_freq threshold are added to the `extra_stop_words` list
    Returns
    -------
    iterable of documents (which is iterable of terms)
        Documents where terms have ben filtered and transformed according to args.

    """

    kwargs = dict(extract_args.get('kwargs', {}))
    args = dict(extract_args.get('args', {}))
    normalize = args.get('normalize', 'lemma')
    substitutions = extract_args.get('substitutions', {})
    extra_stop_words = set(extract_args.get('extra_stop_words', None) or [])
    chunk_size = extract_args.get('chunk_size', None)
    min_length = extract_args.get('min_length', 2)

    # mask_gpe = extract_args.get('mask_gpe', False)
    # if mask_gpe is True:
    #    gpe_names = { x: '_gpe_' for x in get_gpe_names(corpus) }
    #    substitutions = utility.extend(substitutions, gpe_names)

    min_freq = extract_args.get('min_freq', 1)

    if min_freq > 1:
        words = infrequent_words(corpus, normalize=normalize, weighting='count', threshold=min_freq, as_strings=True)
        extra_stop_words = extra_stop_words.union(words)
        logger.info('Ignoring {} low-frequent words!'.format(len(words)))

    max_doc_freq = extract_args.get('max_doc_freq', 100)

    if max_doc_freq < 100:
        words = frequent_document_words(
            corpus, normalize=normalize, weighting='freq', dfs_threshold=max_doc_freq, as_strings=True
        )
        extra_stop_words = extra_stop_words.union(words)
        logger.info('Ignoring {} high-frequent words!'.format(len(words)))

    extract_args = {
        'args': args,
        'kwargs': kwargs,
        'substitutions': substitutions,
        'extra_stop_words': extra_stop_words,
        'chunk_size': chunk_size,
        'min_length': min_length,
    }

    terms = (extract_document_terms(doc, extract_args) for doc in corpus)

    return terms


def chunks(lst, n):
    '''Returns list l in n-sized chunks'''
    if (n or 0) == 0:
        yield lst
    else:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]


def extract_document_tokens(docs, **opts):
    try:
        document_id = 0
        normalize = opts['normalize'] or 'orth'
        term_substitutions = opts.get('substitutions', {})
        min_freq_stats = opts.get('min_freq_stats', {})
        max_doc_freq_stats = opts.get('max_doc_freq_stats', {})
        extra_stop_words = set([])

        if opts['min_freq'] > 1:
            assert normalize in min_freq_stats
            stop_words = utility.extract_counter_items_within_threshold(min_freq_stats[normalize], 1, opts['min_freq'])
            extra_stop_words.update(stop_words)

        if opts['max_doc_freq'] < 100:
            assert normalize in max_doc_freq_stats
            stop_words = utility.extract_counter_items_within_threshold(
                max_doc_freq_stats[normalize], opts['max_doc_freq'], 100
            )
            extra_stop_words.update(stop_words)

        extract_args = dict(
            args=dict(
                ngrams=opts['ngrams'],
                named_entities=opts['named_entities'],
                normalize=opts['normalize'],
                as_strings=True,
            ),
            kwargs=dict(
                min_freq=opts['min_freq'],
                include_pos=opts['include_pos'],
                filter_stops=opts['filter_stops'],
                filter_punct=opts['filter_punct'],
            ),
            extra_stop_words=extra_stop_words,
            substitutions=(term_substitutions if opts.get('substitute_terms', False) else None),
        )

        for document_name, doc in docs:
            # logger.info(document_name)

            terms = [x for x in extract_document_terms(doc, extract_args)]

            chunk_size = opts.get('chunk_size', 0)
            chunk_index = 0
            for tokens in chunks(terms, chunk_size):
                yield document_id, document_name, chunk_index, tokens
                chunk_index += 1

            document_id += 1

    except Exception as ex:
        logger.error(ex)
        raise
