from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Collection, Iterable, List, Literal, Mapping, Optional, Sequence, Union

from loguru import logger

from penelope import utility as pu

from .utils import frequent_document_words, infrequent_words

try:
    from textacy.extract.basics import words

    from ...spacy_api import Doc, Token
except ImportError:
    words = pu.DummyFunction


if TYPE_CHECKING:
    from penelope.corpus import TokensTransformOpts


# FIXME: Deprecate this module, use pipeline instead, spacy => TaggedFrame
# FIXME: PoS-padding (dummy marker) not fully implemented


def to_terms_list(
    doc: Doc,
    *,
    filter_stops: bool = True,
    filter_punct: bool = True,
    filter_nums: bool = False,
    include_pos: Optional[str | Collection[str]] = None,
    exclude_pos: Optional[str | Collection[str]] = None,
    min_freq: int = 1,
    target: Literal['lemma', 'lower', 'text'] = 'lemma',
) -> Iterable[str]:

    tokens: Iterable[Token] = words(
        doc,
        filter_stops=filter_stops,
        filter_punct=filter_punct,
        filter_nums=filter_nums,
        include_pos=include_pos,
        exclude_pos=exclude_pos,
        min_freq=0,
    )

    if target == 'lemma':
        terms: Iterable[str] = (t.lemma_ for t in tokens)
    elif target == 'lower':
        terms: Iterable[str] = (t.lower_ for t in tokens)
    else:
        terms: Iterable[str] = (t.text for t in tokens)

    if min_freq > 1:
        terms = list(tokens)
        frequency: dict = pu.frequencies(terms)
        terms = (t for t in terms if frequency[t] >= min_freq)

    return terms


class ExtractPipeline:
    @dataclass
    class ExtractOpts:
        filter_stops: bool = True
        filter_punct: bool = True
        filter_nums: bool = False
        include_pos: Optional[str | Collection[str]] = None
        exclude_pos: Optional[str | Collection[str]] = None
        min_freq: int = 1

    def __init__(
        self,
        corpus: Iterable[Doc],
        target: Literal['lemma', 'lower', 'text'] = 'lemma',
        tasks: List[Any] = None,
        extract_opts: ExtractOpts = None,
    ):
        self.corpus: Iterable[Doc] = corpus
        self.target: Union[str, Callable] = target
        self.tasks = tasks or []
        self.extract_opts: ExtractPipeline.ExtractOpts = extract_opts or ExtractPipeline.ExtractOpts()

    def add(self, task) -> ExtractPipeline:
        self.tasks.append(task)
        return self

    def process(self) -> Iterable[Iterable[str]]:

        output = self

        for task in self.tasks:
            if hasattr(task, 'setup'):
                output = task.setup(output)

        for doc in self.corpus:

            terms = to_terms_list(doc, **asdict(self.extract_opts))

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
        exclude_pos: Sequence[str] = None,
    ) -> ExtractPipeline:
        return self.add(PoSFilter(include_pos=include_pos, exclude_pos=exclude_pos))

    def ingest_transform_opts(self, transform_opts: TokensTransformOpts) -> ExtractPipeline:

        if not transform_opts.keep_numerals:
            self.ingest(filter_nums=True)

        if transform_opts.remove_stopwords:
            self.remove_stopwords(transform_opts.extra_stopwords)

        if transform_opts.min_len and transform_opts.min_len > 1:
            self.add(MinCharactersFilter(min_length=transform_opts.min_len))

        if transform_opts.max_len:
            self.add(MaxCharactersFilter(max_length=transform_opts.max_len))

        if transform_opts.to_lower:
            self.add(TransformTask(transformer=str.lower))

        if transform_opts.to_upper:
            self.add(TransformTask(transformer=str.upper))

        return self

    def ingest(self, **options) -> ExtractPipeline:
        for k, v in options.items():
            if hasattr(self.extract_opts, k):
                setattr(self.extract_opts, k, v)
            else:
                raise ValueError(f"deprecated option {k}")
                # logger.warning("ignoring unknown option %s", k)
        return self


class StopwordFilter:
    def __init__(self, extra_stopwords=None):
        self.filter_words = extra_stopwords

    def setup(self, pipeline: ExtractPipeline) -> ExtractPipeline:
        pipeline.extract_opts.filter_stops = True
        return pipeline

    def filter(self, terms: Iterable[str]) -> Iterable[str]:
        return (x for x in terms if x not in self.filter_words)


class PoSFilter:
    def __init__(self, include_pos: Sequence[str] = None, exclude_pos: Sequence[str] = None):
        self.include_pos = include_pos
        self.exclude_pos = exclude_pos

        universal_tags = pu.PoS_TAGS_SCHEMES.Universal.tags

        assert all(x in universal_tags for x in (self.include_pos or []))
        assert all(x in universal_tags for x in (self.exclude_pos or []))

    def setup(self, pipeline: ExtractPipeline) -> ExtractPipeline:
        pipeline.extract_opts.include_pos = self.include_pos
        pipeline.extract_opts.exclude_pos = self.exclude_pos
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
                pu.load_term_substitutions(self.filename, default_term='_mask_', delim=';', vocab=self.vocab)
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

        _words = infrequent_words(
            pipeline.corpus,
            normalize=self.target,
            weighting='count',
            threshold=self.min_freq,
            as_strings=True,
        )

        self.filter_words = _words
        logger.info('Ignoring {} low-frequent words!'.format(len(_words)))
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

        _words = frequent_document_words(
            pipeline.corpus,
            normalize=self.target,
            weighting='freq',
            dfs_threshold=self.max_doc_freq,
            as_strings=True,
        )

        self.filter_words = _words
        logger.info('Ignoring {} high-frequent words!'.format(len(_words)))
        return pipeline
