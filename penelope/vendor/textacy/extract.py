from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import penelope.utility as utility
from penelope.corpus import CorpusVectorizer, DocumentIndex, TokensTransformOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts
from spacy.tokens import Doc
from textacy.spacier.doc_extensions import to_terms_list

from .utils import frequent_document_words, infrequent_words, load_term_substitutions

logger = utility.getLogger('corpus_text_analysis')

# FIXME: PoS-replace (dummy marker) not fully implemented


def chunks(lst, n):
    '''Returns list l in n-sized chunks'''
    if (n or 0) == 0:
        yield lst
    else:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]


class ExtractPipeline:
    def __init__(self, corpus: Iterable[Doc], target: Union[str, Callable] = 'lemma', tasks: List[Any] = None):

        self.corpus = corpus
        self.tasks = tasks or []
        self.to_terms_list_args = dict(
            ngrams=1,
            entities=None,
            normalize=target,
            as_strings=True,
        )
        self.to_terms_list_kwargs = dict(
            filter_stops=False,  # (bool)
            filter_punct=True,  # (bool)
            filter_nums=False,  # (bool)
            # POS types
            include_pos=None,  # (str or Set[str])
            exclude_pos=None,  # (str or Set[str])
            # entity types
            include_types=None,  # (str or Set[str])
            exclude_types=None,  # (str or Set[str]
            min_freq=1,  # (int)
            drop_determiners=True,  # (bool)
        )

    def add(self, task) -> ExtractPipeline:
        self.tasks.append(task)
        return self

    def process(self) -> Iterable[Iterable[str]]:

        output = self

        for task in self.tasks:
            if hasattr(task, 'setup'):
                output = task.setup(output)

        for doc in self.corpus:

            terms = to_terms_list(doc, **self.to_terms_list_args, **self.to_terms_list_kwargs)

            for task in self.tasks:
                if hasattr(task, 'apply'):
                    terms = task.apply(terms)

            for task in self.tasks:
                if hasattr(task, 'filter'):
                    terms = task.filter(terms)

            yield list(terms)

    # Short-cuts to predefned tasks:
    def remove_stopwords(self, extra_stopwords=None) -> ExtractPipeline:
        return self.add(StopwordFilter(extra_stopwords=extra_stopwords))

    def ngram(self, ngram: Union[int, Tuple[int]]) -> ExtractPipeline:
        return self.add(NGram(ngram=ngram))

    def ner(
        self,
        include_types: Optional[Union[str, Set[str]]] = None,
        exclude_types: Optional[Union[str, Set[str]]] = None,
        drop_determiners: bool = True,
        min_freq: int = 1,
    ) -> ExtractPipeline:
        return self.add(
            NamedEntityTask(
                include_types=include_types,
                exclude_types=exclude_types,
                drop_determiners=drop_determiners,
                min_freq=min_freq,
            )
        )

    def predicate(self, predicate: Callable[[str], bool]) -> ExtractPipeline:
        return self.add(PredicateFilter(predicate=predicate))

    def transform(self, transformer: Callable[[str], str]) -> ExtractPipeline:
        return self.add(TransformTask(transformer=transformer))

    def substitute(self, subst_map: Mapping[str, str] = None, filename: str = None) -> ExtractPipeline:

        return self.add(SubstitutionTask(subst_map=subst_map, filename=filename))

    def min_character_filter(self, min_length: int = 1) -> ExtractPipeline:
        return self.add(MinCharactersFilter(min_length=min_length))

    def frequent_word_filter(self, max_doc_freq: int = 100, target: str = 'lemma') -> ExtractPipeline:
        return self.add(FrequentWordsFilter(max_doc_freq=max_doc_freq, target=target))

    def infrequent_word_filter(self, min_freq: int = 100, target: str = 'lemma') -> ExtractPipeline:
        return self.add(InfrequentWordsFilter(min_global_count=min_freq, target=target))

    def pos(
        self,
        include_pos: Sequence[str] = None,
        replace_pos: Sequence[str] = None,
        exclude_pos: Sequence[str] = None,
    ) -> ExtractPipeline:
        return self.add(PoSFilter(include_pos=include_pos, replace_pos=replace_pos, exclude_pos=exclude_pos))

    def ingest_transform_opts(self, tokens_transform_opts: TokensTransformOpts) -> ExtractPipeline:

        if not tokens_transform_opts.keep_numerals:
            self.ingest(filter_nums=True)

        if tokens_transform_opts.remove_stopwords:
            self.remove_stopwords(tokens_transform_opts.extra_stopwords)

        if tokens_transform_opts.min_len and tokens_transform_opts.min_len > 1:
            self.add(MinCharactersFilter(min_length=tokens_transform_opts.min_len))

        if tokens_transform_opts.max_len:
            self.add(MaxCharactersFilter(max_length=tokens_transform_opts.max_len))

        if tokens_transform_opts.to_lower:
            self.add(TransformTask(transformer=str.lower))

        if tokens_transform_opts.to_upper:
            self.add(TransformTask(transformer=str.upper))

        # only_alphabetic: bool = False
        # only_any_alphanumeric: bool = False
        # remove_accents: bool = False
        # keep_symbols: bool = True

        return self

    def ingest(self, **options):
        for k, v in options.items():
            if k in self.to_terms_list_args:
                self.to_terms_list_args[k] = v
            elif self.to_terms_list_kwargs:
                self.to_terms_list_kwargs[k] = v
            else:
                logger.warning("ignoring unknown option %s", k)
        return self

    @staticmethod
    def build(corpus: Iterable[Doc], target: str, **options):
        pipeline = ExtractPipeline(corpus, target).ingest(**options)
        return pipeline


class StopwordFilter:
    def __init__(self, extra_stopwords=None):
        self.filter_words = extra_stopwords

    def setup(self, pipeline: ExtractPipeline) -> ExtractPipeline:
        pipeline.to_terms_list_kwargs['filter_stops'] = True
        return pipeline

    def filter(self, terms: Iterable[str]) -> Iterable[str]:
        return (x for x in terms if x not in self.filter_words)


class PoSFilter:
    def __init__(
        self, include_pos: Sequence[str] = None, replace_pos: Sequence[str] = None, exclude_pos: Sequence[str] = None
    ):
        self.include_pos = include_pos
        self.replace_pos = replace_pos
        self.exclude_pos = exclude_pos

        universal_tags = utility.PoS_TAGS_SCHEMES.Universal.tags

        assert all([x in universal_tags for x in (self.include_pos or [])])
        assert all([x in universal_tags for x in (self.exclude_pos or [])])
        assert all([x in universal_tags for x in (self.replace_pos or [])])

    def setup(self, pipeline: ExtractPipeline) -> ExtractPipeline:
        pipeline.to_terms_list_kwargs['include_pos'] = self.include_pos
        pipeline.to_terms_list_kwargs['replace_pos'] = self.replace_pos
        pipeline.to_terms_list_kwargs['exclude_pos'] = self.exclude_pos
        return pipeline


class NGram:
    def __init__(self, ngram=1):
        self.ngram = ngram

    def setup(self, pipeline: ExtractPipeline) -> ExtractPipeline:
        pipeline.to_terms_list_kwargs['ngram'] = self.ngram
        return pipeline


class NamedEntityTask:
    def __init__(
        self,
        include_types: Optional[Union[str, Set[str]]] = None,
        exclude_types: Optional[Union[str, Set[str]]] = None,
        drop_determiners: bool = True,
        min_freq: int = 1,
    ):
        self.include_types = include_types
        self.exclude_types = exclude_types
        self.drop_determiners = drop_determiners
        self.min_freq = min_freq

    def setup(self, pipeline: ExtractPipeline):
        pipeline.to_terms_list_args['entities'] = True
        pipeline.to_terms_list_kwargs['include_types'] = self.include_types
        pipeline.to_terms_list_kwargs['exclude_types'] = self.exclude_types
        pipeline.to_terms_list_kwargs['drop_determiners'] = self.drop_determiners
        pipeline.to_terms_list_kwargs['min_freq'] = self.min_freq
        return pipeline


class ExtractOptions:
    def __init__(self, **options):
        self.options = options

    def setup(self, pipeline: ExtractPipeline) -> ExtractPipeline:
        pipeline.ingest(**self.options)
        return pipeline


class PredicateFilter:
    def __init__(self, predicate: Callable[[str], bool]):
        self.predicate = predicate

    def filter(self, terms: Iterable[str]) -> Iterable[str]:
        return (x for x in terms if self.predicate(x))


class TransformTask:
    def __init__(self, transformer: Callable[[str], bool]):
        self.transformer = transformer

    def apply(self, terms: Iterable[str]) -> Iterable[str]:
        return (self.transformer(x) for x in terms)


class SubstitutionTask:
    def __init__(self, subst_map: Mapping[str, str] = None, filename: str = None, vocab=None):
        self.subst_map = subst_map
        self.filename = filename
        self.vocab = vocab

    def setup(self, pipeline: ExtractPipeline) -> ExtractPipeline:

        if self.filename is not None:

            self.subst_map = self.subst_map or {}

            if not os.path.isfile(self.filename):
                raise FileNotFoundError(f"terms substitions file {self.filename} not found")

            self.subst_map.update(
                load_term_substitutions(self.filename, default_term='_mask_', delim=';', vocab=self.vocab)
            )

        return pipeline

    def apply(self, terms: Iterable[str]) -> Iterable[str]:
        return (self.subst_map[x] if x in self.subst_map else x for x in terms)


class MinCharactersFilter(PredicateFilter):
    def __init__(self, min_length: int = 2):
        super().__init__(predicate=lambda x: len(x) >= min_length)


class MaxCharactersFilter(PredicateFilter):
    def __init__(self, max_length: int = 100):
        super().__init__(predicate=lambda x: len(x) <= max_length)


class InfrequentWordsFilter(StopwordFilter):
    def __init__(self, min_global_count: int = 100, target: str = 'lemma'):
        self.min_freq = min_global_count
        self.target = target
        super().__init__(extra_stopwords=[])

    def setup(self, pipeline: ExtractPipeline) -> ExtractPipeline:

        words = infrequent_words(
            pipeline.corpus,
            normalize=self.target,
            weighting='count',
            threshold=self.min_freq,
            as_strings=True,
        )

        self.filter_words = words
        logger.info('Ignoring {} low-frequent words!'.format(len(words)))
        return pipeline


class FrequentWordsFilter(StopwordFilter):
    def __init__(self, max_doc_freq: int = 100, target: str = 'lemma'):
        self.max_doc_freq = max_doc_freq
        self.target = target
        super().__init__(extra_stopwords=[])

    def setup(self, pipeline: ExtractPipeline) -> ExtractPipeline:

        if self.max_doc_freq >= 100:
            logger.info("max document frequency filter ignored (value >= 100 not allowed")
            return pipeline

        words = frequent_document_words(
            pipeline.corpus,
            normalize=self.target,
            weighting='freq',
            dfs_threshold=self.max_doc_freq,
            as_strings=True,
        )

        self.filter_words = words
        logger.info('Ignoring {} high-frequent words!'.format(len(words)))
        return pipeline


def extract_document_tokens(
    *,
    spacy_docs: Iterable[Doc],
    document_index: DocumentIndex,
    extract_tokens_opts: ExtractTaggedTokensOpts = None,
    tokens_transform_opts: TokensTransformOpts = None,
    extract_args: Dict[str, Any] = None,
) -> Iterable[Tuple[str, Iterable[str]]]:

    target = "lemma" if extract_tokens_opts.lemmatize else "text"
    # FIXME: Implement pos_paddings
    tokens_stream = (
        ExtractPipeline.build(corpus=spacy_docs, target=target)
        .pos(
            include_pos=extract_tokens_opts.pos_includes,
            replace_pos=extract_tokens_opts.replace_pos,
            exclude_pos=extract_tokens_opts.pos_excludes,
        )
        .ingest_transform_opts(tokens_transform_opts)
        .ingest(**extract_args)
        .process()
    )
    document_tokens = zip(document_index.filename, tokens_stream)

    return document_tokens


def vectorize_textacy_corpus(
    *,
    spacy_docs: Iterable[Doc],
    document_index: DocumentIndex,
    extract_tokens_opts: ExtractTaggedTokensOpts = None,
    tokens_transform_opts: TokensTransformOpts = None,
    extract_args: Dict[str, Any] = None,
    vectorizer_args=None,
):
    document_tokens = extract_document_tokens(
        spacy_docs=spacy_docs,
        document_index=document_index,
        extract_tokens_opts=extract_tokens_opts,
        tokens_transform_opts=tokens_transform_opts,
        extract_args=extract_args,
    )

    v_corpus = CorpusVectorizer().fit_transform(
        corpus=document_tokens,
        document_index=document_index,
        verbose=True,
        **{
            **vectorizer_args,
        },
    )

    return v_corpus
