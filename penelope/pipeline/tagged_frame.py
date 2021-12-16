import glob
from dataclasses import dataclass, field
from enum import IntEnum
from os.path import join as jj
from typing import Iterable, List, Mapping

import numpy as np
import pandas as pd
from penelope.corpus import DocumentIndexHelper, Token2Id
from tqdm import tqdm

from ..utility import PoS_Tag_Scheme
from .interfaces import ITask
from .pipeline import ContentType, DocumentPayload
from .tasks import Vocabulary
from .tasks_mixin import CountTaggedTokensMixIn, DefaultResolveMixIn


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

        id_tagged_frame: pd.DataFrame = pd.DataFrame(
            data=dict(
                token_id=tagged_frame[token_column].map(self.token2id).astype(np.int32),
                pos_id=tagged_frame[pos_column].map(pos_schema.pos_to_id).astype(np.int8),
            )
        )
        return payload.update(ContentType.TAGGED_ID_FRAME, id_tagged_frame)


# Copied from westac.parlaclarin.tasks


@dataclass
class LoadGroupedIdTaggedFrame(CountTaggedTokensMixIn, DefaultResolveMixIn, ITask):
    """Load numerical tagged frames in stored CSV or FEATHER format"""

    # source_folder: str = '/data/riksdagen_corpus_data/tagged-speech-corpus.numeric.feather'
    corpus_source: str = ""
    file_pattern: str = "**/*.feather"
    to_tagged_frame: bool = False
    vocabulary: pd.DataFrame = None
    token2id: Token2Id = None
    document_index: pd.DataFrame = None
    document_id2name: Mapping[int, str] = None
    document_tfs: dict = None

    pos_frequency_column: str = None
    lemma_frequency_column: str = None

    def __post_init__(self):
        self.in_content_type = ContentType.NONE
        self.out_content_type = ContentType.TAGGED_FRAME if self.to_tagged_frame else ContentType.TAGGED_ID_FRAME

        if not self.file_pattern.endswith('.feather'):
            raise ValueError("Only feather files are currently supported")

    def enter(self) -> None:
        super().enter()

        if self.corpus_source is None:
            raise FileNotFoundError("LoadTaggedFrame: Corpus source is None")

        self.vocabulary = self.read_vocabulary()

        self.vocabulary['token'] = self.vocabulary['token'].fillna('')

        self.document_index: pd.DataFrame = DocumentIndexHelper.load(
            jj(self.corpus_source, 'document_index.feather')
        ).document_index

        self.document_id2name: Mapping[int, str] = {
            doc_id: doc_name
            for doc_id, doc_name in zip(self.document_index.document_id, self.document_index.document_name)
        }

        # self.document_id2name: Mapping[int, str] = self.document_index.set_index('document_id')[
        #     'document_name'
        # ].to_dict()

        # self.token2id: Token2Id = Token2Id(data=self.vocabulary.set_index('token').token_id.to_dict())
        self.token2id: Token2Id = Token2Id(data={t: i for t, i in zip(self.vocabulary.token, self.vocabulary.token_id)})

        self.pipeline.payload.effective_document_index = self.document_index
        self.pipeline.payload.token2id = self.token2id
        self.document_tfs = {}

    # def exit(self):
    #     super().exit()
    # with contextlib.suppress(Exception):
    #     with open(jj(self.corpus_source, 'document_tfs.json'), "w", encoding='utf-8') as fp:
    #         json.dump(self.document_tfs, fp)

    def read_vocabulary(self):
        """Read vocabulary. Return data frame."""
        return pd.read_feather(jj(self.corpus_source, 'token2id.feather'))

    def create_instream(self) -> Iterable[DocumentPayload]:

        filenames: List[str] = self.corpus_filenames()

        fg = self.token2id.id2token.get
        dg = self.document_id2name.get
        pg = self.pipeline.payload.pos_schema.id_to_pos.get

        text_column, pos_column, lemma_column = self.pipeline.payload.tagged_columns_names2
        # term_frequency_name, pos_frequency_name = (
        #     ('lemma_id', 'pos_id') if not self.to_tagged_frame else (lemma_column, pos_column)
        # )

        for filename in tqdm(filenames, total=len(filenames)):
            # document_name: str = utility.strip_path_and_extension(filename)
            group_frame: pd.DataFrame = self.load_tagged_frame(filename)

            if self.to_tagged_frame:

                if 'token_id' in group_frame.columns:
                    group_frame[text_column] = group_frame.token_id.apply(fg)

                if 'lemma_id' in group_frame.columns:
                    group_frame[lemma_column] = group_frame.lemma_id.apply(fg)

                group_frame[pos_column] = group_frame.pos_id.apply(pg)

                # group_frame.update(tagged_frame[tagged_frame.lemma_id.isna()].token_id)

                group_frame.drop(columns=['token_id', 'pos_id', 'lemma_id'], inplace=True, errors='ignore')

            for document_id, tagged_frame in group_frame.groupby('document_id'):

                tagged_frame.reset_index(drop=True, inplace=True)

                payload: DocumentPayload = DocumentPayload(
                    content_type=self.out_content_type,
                    content=tagged_frame,
                    filename=dg(document_id),
                )
                # tfs: dict = dict(
                #     term_frequency=term_frequency(tagged_frame[term_frequency_name]),
                #     pos_frequency=term_frequency(tagged_frame[pos_frequency_name]),
                # )
                # self.document_tfs[document_id] = tfs
                yield payload

    def load_tagged_frame(self, filename) -> pd.DataFrame:
        tagged_frame: pd.DataFrame = pd.read_feather(filename)
        # tagged_frame.update(tagged_frame[tagged_frame.lemma_id.isna()].token_id)
        return tagged_frame

    def corpus_filenames(self) -> List[str]:
        return sorted(glob.glob(jj(self.corpus_source, self.file_pattern), recursive=True))
