from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import pandas as pd
import penelope.utility as utility
import textacy
from penelope.corpus import VectorizedCorpus
from sklearn.feature_extraction.text import CountVectorizer

from .tags import PoS_tags
from .utils import frequent_document_words, infrequent_words, load_term_substitutions

logger = utility.getLogger('corpus_text_analysis')


def chunks(lst, n):
    '''Returns list l in n-sized chunks'''
    if (n or 0) == 0:
        yield lst
    else:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]


class ExtractPipeline:
    def __init__(self, corpus: textacy.Corpus, target: Union[str, Callable] = 'lemma', tasks: List[Any] = None):

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

            terms = doc._.to_terms_list(**self.to_terms_list_args, **self.to_terms_list_kwargs)

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

    def pos(self, include_pos: Sequence[str] = None, exclude_pos: Sequence[str] = None) -> ExtractPipeline:
        return self.add(PoSFilter(include_pos=include_pos, exclude_pos=exclude_pos))

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
    def build(corpus, target, **options):
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
    def __init__(self, include_pos: Sequence[str] = None, exclude_pos: Sequence[str] = None):
        self.include_pos = include_pos
        self.exclude_pos = exclude_pos

        assert all([x in PoS_tags for x in (self.include_pos or [])])
        assert all([x in PoS_tags for x in (self.exclude_pos or [])])

    def setup(self, pipeline: ExtractPipeline) -> ExtractPipeline:
        pipeline.to_terms_list_kwargs['include_pos'] = self.include_pos
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


def vectorize_textacy_corpus(
    corpus,
    documents: pd.DataFrame,
    n_count: int,
    n_top: int,
    normalize_axis=None,
    year_range: Tuple[int, int] = (1920, 2020),
    extract_args: Dict[str, Any] = None,
    vecargs=None,
):

    target = extract_args.get("normalize", "lemma")

    document_stream = ExtractPipeline.build(corpus, target).ingest(**extract_args).process()

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), **(vecargs or {}))

    bag_term_matrix = vectorizer.fit_transform(document_stream)

    x_corpus = VectorizedCorpus(bag_term_matrix, vectorizer.vocabulary_, documents)

    year_range = (
        x_corpus.documents.year.min(),
        x_corpus.documents.year.max(),
    )
    year_filter = lambda x: year_range[0] <= x["year"] <= year_range[1]

    x_corpus = x_corpus.filter(year_filter).group_by_year().slice_by_n_count(n_count).slice_by_n_top(n_top)

    for axis in normalize_axis or []:
        x_corpus = x_corpus.normalize(axis=axis, keep_magnitude=False)

    return x_corpus
