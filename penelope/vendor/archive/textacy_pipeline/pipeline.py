# from __future__ import annotations

# import abc
# import os
# import zipfile
# from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Union

# import pandas as pd
# from loguru import logger
# from spacy.language import Language as SpacyLanguage
# from spacy.tokens import Doc as SpacyDoc
# from textacy.corpus import Corpus
# from textacy.spacier.utils import make_doc_from_text_chunks
# from tqdm.auto import tqdm

# from penelope.utility import (
#     faster_to_dict_records,
#     lists_of_dicts_merged_by_key,
#     noop,
#     path_add_suffix,
#     streamify_any_source,
# )

# from ..spacy import load_model_by_parts
# from ..textacy_api import load_corpus

# if TYPE_CHECKING:
#     from penelope.corpus.readers import ICorpusReader, TextReaderOpts, TextTransformer


# class PipelineError(Exception):
#     pass


# class TextacyCorpusPipeline:
#     def __init__(
#         self,
#         *,
#         filename: str,
#         lang: str,
#         document_index: pd.DataFrame,
#         tasks: Sequence[ITask] = None,
#         disables: str = "ner,",
#         force: bool = False,
#         reader_opts: TextReaderOpts,
#         # FIXME: Move to caller; transformer = TextTransformer().fix_hyphenation().fix_unicode().fix_whitespaces().fix_ftfy()
#         transformer: TextTransformer,
#     ):

#         self.filename = filename
#         self._tasks: List[ITask] = tasks or []
#         self.document_index = document_index
#         self.lang = lang
#         self.nlp: SpacyLanguage = load_model_by_parts(lang=lang, disable=tuple(disables.split(',')))
#         self.corpus: Corpus = None
#         self.force = force
#         self.suffix = '_preprocessed'
#         self.reader_opts = reader_opts
#         self.transformer: TextTransformer = transformer

#     def process(self) -> TextacyCorpusPipeline:
#         for task in self._tasks:
#             task().execute(self)
#         return self

#     def add(self, task: Union[ITask, List[ITask]]) -> TextacyCorpusPipeline:
#         self._tasks.extend([task] if isinstance(task, ITask) else task)
#         return self

#     def create(self) -> TextacyCorpusPipeline:
#         return self.add(CreateTask)

#     def preprocess(self) -> TextacyCorpusPipeline:
#         return self.add(PreprocessTask)

#     def save(self) -> TextacyCorpusPipeline:
#         return self.add(SaveTask)

#     def load(self) -> TextacyCorpusPipeline:
#         return self.add(LoadTask)


# class ITask(abc.ABC):
#     @abc.abstractmethod
#     def execute(self, pipeline: TextacyCorpusPipeline) -> TextacyCorpusPipeline:
#         return None


# class CreateTask(ITask):
#     def execute(self, pipeline: TextacyCorpusPipeline, stream: ICorpusReader):
#         extra_metadata = faster_to_dict_records(pipeline.document_index)
#         # extra_metadata = pipeline.document_index.to_dict('records')
#         pipeline.corpus = self.create_corpus(stream, pipeline.nlp, extra_metadata=extra_metadata)
#         return pipeline

#     def create_corpus(
#         self,
#         reader: ICorpusReader,
#         nlp: SpacyLanguage,
#         *,
#         extra_metadata: List[Dict[str, Any]] = None,
#         tick: Callable = noop,
#         n_chunk_threshold: int = 100000,
#     ) -> Corpus:

#         corpus: Corpus = Corpus(nlp)
#         counter = 0

#         metadata_mapping = {
#             x['filename']: x for x in lists_of_dicts_merged_by_key(reader.metadata, extra_metadata, key='filename')
#         }

#         for filename, text in reader:

#             metadata = metadata_mapping[filename]

#             if len(text) > n_chunk_threshold:
#                 doc: SpacyDoc = make_doc_from_text_chunks(text, lang=nlp, chunk_size=n_chunk_threshold)
#                 corpus.add_doc(doc)
#                 doc._.meta = metadata
#             else:
#                 corpus.add((text, metadata))

#             counter += 1
#             if counter % 100 == 0:
#                 logger.info('%s documents added...', counter)

#             tick(counter)

#         return corpus


# class PreprocessTask(ITask):
#     def preprocess(
#         self,
#         transformer: TextTransformer,
#         source_filename: str,
#         target_filename: str,
#         filename_pattern: str = '*.txt',
#         _tqdm=tqdm,
#     ):
#         """Creates a preprocessed version of an archive"""

#         source = streamify_any_source(source_filename, filename_pattern=filename_pattern)
#         if _tqdm is not None:
#             source = _tqdm(source, desc='Preparing text corpus')

#         with zipfile.ZipFile(target_filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
#             for filename, text in source:
#                 text = transformer.transform(text)
#                 zf.writestr(filename, text)

#     def execute(self, pipeline: TextacyCorpusPipeline):
#         target_filename = path_add_suffix(pipeline.filename, pipeline.suffix)
#         if not os.path.isfile(target_filename) or pipeline.force:
#             self.preprocess(pipeline.filename, target_filename)
#         pipeline.filename = target_filename
#         return pipeline


# class SaveLoadMixIn:
#     def generate_corpus_filename(
#         self,
#         source_path: str,
#         language: str,
#         nlp_args=None,
#         preprocess_args=None,
#         compression: str = 'bz2',
#         extension: str = 'bin',
#     ) -> str:
#         nlp_args = nlp_args or {}
#         preprocess_args = preprocess_args or {}
#         disabled_pipes = nlp_args.get('disable', ())
#         suffix = '_{}_{}{}'.format(
#             language,
#             '_'.join([k for k in preprocess_args if preprocess_args[k]]),
#             '_disable({})'.format(','.join(disabled_pipes)) if len(disabled_pipes) > 0 else '',
#         )
#         filename = path_add_suffix(source_path, suffix, new_extension='.' + extension)
#         if (compression or '') != '':
#             filename += '.' + compression
#         return filename


# class SaveTask(SaveLoadMixIn, ITask):
#     def execute(self, pipeline: TextacyCorpusPipeline):
#         if pipeline.corpus is None:
#             raise PipelineError("save when corpus is None")
#         textacy_corpus_path = self.generate_corpus_filename(pipeline.filename, pipeline.lang)
#         if os.path.isfile(textacy_corpus_path) or pipeline.force:
#             pipeline.corpus.save(textacy_corpus_path)
#         return pipeline


# class LoadTask(SaveLoadMixIn, ITask):
#     def execute(self, pipeline: TextacyCorpusPipeline):
#         if pipeline.corpus is None:
#             textacy_corpus_path = self.generate_corpus_filename(pipeline.filename, pipeline.lang)
#             pipeline.corpus = load_corpus(textacy_corpus_path, pipeline.nlp)
#         return pipeline
