from __future__ import annotations

import abc
import os
from typing import List, Sequence, Union

import pandas as pd
import penelope.vendor.textacy as textacy_utility
import textacy
from penelope.corpus import preprocess_text_corpus
from penelope.corpus.readers import ZipTextIterator
from penelope.utility import IndexOfSplitOrCallableOrRegExp, path_add_suffix
from spacy.language import Language as SpacyLanguage


class PipelineError(Exception):
    pass


class TextacyCorpusPipeline:
    def __init__(
        self,
        *,
        filename: str,
        lang: str,
        documents: pd.DataFrame,
        tasks: Sequence[ITask] = None,
        disables: str = "ner,",
        force: bool = False,
        filename_pattern: str = '*.txt',
        filename_fields: List[IndexOfSplitOrCallableOrRegExp],
    ):

        self.filename = filename
        self._tasks: List[ITask] = tasks or []
        self.documents = documents
        self.lang = lang
        self.nlp: SpacyLanguage = textacy_utility.create_nlp(lang, disable=tuple(disables.split(',')))
        self.corpus: textacy.Corpus = None
        self.force = force
        self.suffix = '_preprocessed'
        self.filename_pattern = filename_pattern
        self.filename_fields = filename_fields

    def process(self) -> TextacyCorpusPipeline:
        for task in self._tasks:
            task().execute(self)
        return self

    def add(self, task: Union[ITask, List[ITask]]) -> TextacyCorpusPipeline:
        self._tasks.extend([task] if isinstance(task, ITask) else task)
        return self

    def create(self) -> TextacyCorpusPipeline:
        return self.add(CreateTask)

    def preprocess(self) -> TextacyCorpusPipeline:
        return self.add(PreprocessTask)

    def save(self) -> TextacyCorpusPipeline:
        return self.add(SaveTask)

    def load(self) -> TextacyCorpusPipeline:
        return self.add(LoadTask)


class ITask(abc.ABC):
    @abc.abstractmethod
    def execute(self, pipeline: TextacyCorpusPipeline) -> TextacyCorpusPipeline:
        return None


class CreateTask(ITask):
    def execute(self, pipeline: TextacyCorpusPipeline):
        stream = ZipTextIterator(
            pipeline.filename, filename_pattern=pipeline.filename_pattern, filename_fields=pipeline.filename_fields
        )
        extra_metadata = pipeline.documents.to_dict('records')
        pipeline.corpus = textacy_utility.create_corpus(stream, pipeline.nlp, extra_metadata=extra_metadata)
        return pipeline


class PreprocessTask(ITask):
    def execute(self, pipeline: TextacyCorpusPipeline):
        prepped_source_path = path_add_suffix(pipeline.filename, pipeline.suffix)
        if not os.path.isfile(prepped_source_path) or pipeline.force:
            preprocess_text_corpus(pipeline.filename, prepped_source_path)
        pipeline.filename = prepped_source_path
        return pipeline


class SaveTask(ITask):
    def execute(self, pipeline: TextacyCorpusPipeline):
        if pipeline.corpus is None:
            raise PipelineError("save when corpus is None")
        textacy_corpus_path = textacy_utility.generate_corpus_filename(pipeline.filename, pipeline.lang)
        if os.path.isfile(textacy_corpus_path) or pipeline.force:
            pipeline.corpus.save(textacy_corpus_path)
        return pipeline


class LoadTask(ITask):
    def execute(self, pipeline: TextacyCorpusPipeline):
        if pipeline.corpus is None:
            textacy_corpus_path = textacy_utility.generate_corpus_filename(pipeline.filename, pipeline.lang)
            pipeline.corpus = textacy_utility.load_corpus(textacy_corpus_path, pipeline.nlp)
        return pipeline
