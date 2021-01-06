from __future__ import annotations

import abc
import os
import zipfile
from typing import List, Sequence, Union

import pandas as pd
import penelope.vendor.textacy as textacy_utility
import textacy
from penelope.corpus.readers import TextReaderOpts, TextTransformer, ZipTextIterator
from penelope.utility import create_iterator, path_add_suffix
from spacy.language import Language as SpacyLanguage
from tqdm.auto import tqdm


class PipelineError(Exception):
    pass


class TextacyCorpusPipeline:
    def __init__(
        self,
        *,
        filename: str,
        lang: str,
        document_index: pd.DataFrame,
        tasks: Sequence[ITask] = None,
        disables: str = "ner,",
        force: bool = False,
        reader_opts: TextReaderOpts,
    ):

        self.filename = filename
        self._tasks: List[ITask] = tasks or []
        self.document_index = document_index
        self.lang = lang
        self.nlp: SpacyLanguage = textacy_utility.create_nlp(lang, disable=tuple(disables.split(',')))
        self.corpus: textacy.Corpus = None
        self.force = force
        self.suffix = '_preprocessed'
        self.reader_opts = reader_opts

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
            pipeline.filename,
            reader_opts=pipeline.reader_opts,
        )
        extra_metadata = pipeline.document_index.to_dict('records')
        pipeline.corpus = textacy_utility.create_corpus(stream, pipeline.nlp, extra_metadata=extra_metadata)
        return pipeline


class PreprocessTask(ITask):
    def preprocess(self, source_filename: str, target_filename: str, filename_pattern: str = '*.txt', _tqdm=tqdm):
        """Creates a preprocessed version of an archive"""

        transformer = TextTransformer().fix_hyphenation().fix_unicode().fix_whitespaces().fix_ftfy()
        source = create_iterator(source_filename, filename_pattern=filename_pattern)
        if _tqdm is not None:
            source = _tqdm(source, desc='Preparing text corpus')

        with zipfile.ZipFile(target_filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            for filename, text in source:
                text = transformer.transform(text)
                zf.writestr(filename, text)

    def execute(self, pipeline: TextacyCorpusPipeline):
        target_filename = path_add_suffix(pipeline.filename, pipeline.suffix)
        if not os.path.isfile(target_filename) or pipeline.force:
            self.preprocess(pipeline.filename, target_filename)
        pipeline.filename = target_filename
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
