from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from enum import IntEnum
from fnmatch import fnmatch
from functools import cached_property
from os.path import join as jj
from typing import Iterable, List, Mapping

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from penelope import corpus as pc
from penelope.utility import PoS_Tag_Scheme, replace_extension, strip_paths

from .interfaces import ITask
from .pipeline import ContentType, DocumentPayload
from .tasks import Vocabulary
from .tasks_mixin import DefaultResolveMixIn, PoSCountMixIn


class IngestVocabType(IntEnum):
    """Use supplied vocab"""

    Supplied = 0
    """Build vocab in separate pass on enter"""
    Prebuild = 1
    """Ingest tokens from each document incrementally"""
    Incremental = 2


@dataclass
class ToIdTaggedFrame(Vocabulary):
    """Encode a TaggedFrame to a numerical representation.
    Return data frame with columns `token_id` and `pos_id`.
    """

    ingest_vocab_type: str = field(default=IngestVocabType.Incremental)

    def __post_init__(self):

        super().__post_init__()

        self.in_content_type = ContentType.TAGGED_FRAME
        self.out_content_type = ContentType.TAGGED_ID_FRAME

    def setup(self) -> ITask:

        if self.ingest_vocab_type == IngestVocabType.Supplied:

            self.token2id = self.token2id or self.pipeline.payload.token2id

            if self.token2id is None or len(self.token2id) == 0:
                raise ValueError("Non-empty token2id must be supplied when ingest type is Supplied")

            if self.pipeline.payload.token2id is not self.token2id:
                self.pipeline.payload.token2id = self.token2id

        return super().setup()

    def enter(self):

        if self.ingest_vocab_type == IngestVocabType.Prebuild:
            super().enter()

        if self.ingest_vocab_type == IngestVocabType.Incremental:
            self.token2id.ingest(self.token2id.magic_tokens)

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

        tagged_frame: pd.DataFrame = payload.content

        pos_schema: PoS_Tag_Scheme = self.pipeline.config.pipeline_payload.pos_schema
        token_column: str = self.target
        pos_column: str = self.pipeline.get('pos_column', None)

        if self.ingest_vocab_type == IngestVocabType.Incremental:
            self.token2id.ingest(tagged_frame[token_column])  # type: ignore

        fg = self.token2id.data.get
        pg = pos_schema.pos_to_id.get

        id_tagged_frame: pd.DataFrame = pd.DataFrame(
            data=dict(
                token_id=tagged_frame[token_column].apply(fg).astype(np.int32),
                pos_id=tagged_frame[pos_column].apply(pg).astype(np.int8),
            )
        )
        return payload.update(ContentType.TAGGED_ID_FRAME, id_tagged_frame)


@dataclass
class StoreIdTaggedFrame(ITask):
    """Stores numerical tagged frames stored in FEATHER format.
    Each tagged frame can contain several document identified by a 'document_id' column
    """

    folder: str = ""

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGED_ID_FRAME
        self.out_content_type = ContentType.TAGGED_ID_FRAME

    def enter(self) -> None:
        super().enter()
        os.makedirs(self.folder, exist_ok=True)

    def exit(self) -> None:
        super().exit()
        self.pipeline.payload.document_index.reset_index().to_feather(jj(self.folder, 'document_index.feather'))
        self.pipeline.payload.token2id.to_feather(jj(self.folder, 'token2id.feather'))

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:
        payload = super().process_payload(payload)
        payload.content.to_feather(jj(self.folder, replace_extension(payload.filename, 'feather')))
        return payload


@dataclass
class LoadIdTaggedFrame(PoSCountMixIn, DefaultResolveMixIn, ITask):
    """Loads numerical tagged frames stored in CSV or FEATHER format.
    Each tagged frame can contain several document identified by a 'document_id' column
    """

    corpus_source: str = ""
    file_pattern: str = "**/*.feather"
    id_to_token: bool = False

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TAGGED_FRAME if self.id_to_token else ContentType.TAGGED_ID_FRAME
        if not self.file_pattern.endswith('.feather'):
            raise ValueError("Only feather files are currently supported")

    def setup(self) -> ITask:
        super().setup()
        if self.corpus_source is None:
            raise FileNotFoundError("LoadTaggedFrame: Corpus source is None")
        self.pipeline.payload.effective_document_index = self.document_index
        self.pipeline.payload.token2id = self.token2id
        return self

    @cached_property
    def vocabulary(self) -> pd.DataFrame:
        vocab: pd.DataFrame = pd.read_feather(jj(self.corpus_source, 'token2id.feather'))
        vocab.token.fillna('', inplace=True)
        return vocab

    @cached_property
    def token2id(self) -> pc.Token2Id:
        return pc.Token2Id(data={t: i for t, i in zip(self.vocabulary.token, self.vocabulary.token_id)})

    @cached_property
    def document_index(self) -> pd.DataFrame:
        return pc.DocumentIndexHelper.load(jj(self.corpus_source, 'document_index.feather')).document_index

    @cached_property
    def docid2name(self) -> Mapping[int, str]:
        return self.document_index.set_index('document_id')['document_name'].to_dict()

    def create_instream(self) -> Iterable[DocumentPayload]:

        fg = self.token2id.id2token.get
        dg = self.docid2name.get
        pg = self.pipeline.payload.pos_schema.id_to_pos.get

        text_column, pos_column, lemma_column = self.pipeline.payload.tagged_columns_names2

        loaded_frame_columns: set = None

        for filename in tqdm(self.corpus_filenames, total=len(self.corpus_filenames)):

            loaded_frame: pd.DataFrame = self.load_tagged_frame(filename)

            loaded_frame_columns = loaded_frame_columns or set(loaded_frame.columns)

            if loaded_frame_columns != set(loaded_frame.columns):
                raise ValueError(
                    f"columns in tagged frames {filename} differs from previous frame (is correct file pattern set?)"
                )

            if self.id_to_token:

                if 'token_id' in loaded_frame.columns:
                    loaded_frame[text_column] = loaded_frame.token_id.apply(fg)

                if 'lemma_id' in loaded_frame.columns:
                    loaded_frame[lemma_column] = loaded_frame.lemma_id.apply(fg)

                loaded_frame[pos_column] = loaded_frame.pos_id.apply(pg)

                loaded_frame.drop(columns=['token_id', 'pos_id', 'lemma_id'], inplace=True, errors='ignore')

            if 'document_id' not in loaded_frame_columns:

                payload: DocumentPayload = DocumentPayload(
                    content_type=self.out_content_type, content=loaded_frame, filename=strip_paths(filename)
                )
                self.register_pos_counts(payload)

                yield payload

            else:

                for document_id, tagged_frame in loaded_frame.groupby('document_id'):

                    tagged_frame.reset_index(drop=True, inplace=True)
                    payload: DocumentPayload = DocumentPayload(
                        content_type=self.out_content_type, content=tagged_frame, filename=dg(document_id)
                    )
                    self.register_pos_counts(payload)

                    yield payload

    def load_tagged_frame(self, filename) -> pd.DataFrame:
        tagged_frame: pd.DataFrame = pd.read_feather(filename)
        return tagged_frame

    @cached_property
    def corpus_filenames(self) -> List[str]:
        magic_names: List[str] = ['token2id*feather', 'document_index*feather']
        match_iter: Iterable[str] = (
            filename
            for filename in glob.iglob(jj(self.corpus_source, self.file_pattern), recursive=True)
            if not any(fnmatch(strip_paths(filename), m) for m in magic_names)
        )

        filenames: List[str] = sorted(match_iter)

        logger.info(f"found {len(filenames)} matching `{self.file_pattern}` in {self.corpus_source}")

        if len(filenames) == 0:
            raise FileNotFoundError("no files found in source")

        return filenames
