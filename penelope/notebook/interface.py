import os
from dataclasses import dataclass
from typing import List

from penelope.co_occurrence.interface import ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts
from penelope.corpus.dtm.vectorizer import VectorizeOpts
from penelope.pipeline import CorpusType
from penelope.utility import PropertyValueMaskingOpts

# pylint: disable=too-many-instance-attributes


class WorkflowException(Exception):
    pass


@dataclass
class ComputeOpts:

    corpus_type: CorpusType
    corpus_filename: str
    target_folder: str
    corpus_tag: str
    tokens_transform_opts: TokensTransformOpts
    text_reader_opts: TextReaderOpts
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts
    tagged_tokens_filter_opts: PropertyValueMaskingOpts
    vectorize_opts: VectorizeOpts
    count_threshold: int
    create_subfolder: bool
    persist: bool
    force: bool = False

    context_opts: ContextOpts = None
    partition_keys: List[str] = None

    def is_satisfied(self):

        if not self.corpus_filename:
            raise ValueError("please specify corpus file")

        if not os.path.isfile(self.corpus_filename):
            raise FileNotFoundError(self.corpus_filename)

        if not self.corpus_tag:
            raise ValueError("please specify output tag")

        if not self.target_folder:
            raise ValueError("please specify output folder")

        if len(self.text_reader_opts.filename_fields or []) == 0:
            raise ValueError("please specify at least one filename field")

        if self.context_opts:

            # if len(self.context_opts.concept or []) == 0:
            #     raise ValueError("please specify at least one concept")

            if self.context_opts.context_width is None:
                raise ValueError("please specify at width of context as max distance from concept")

            if len(self.partition_keys or []) == 0:
                raise ValueError("please specify partition key")

            if len(self.partition_keys) > 1:
                raise ValueError("only one partition key is allowed (for now)")

        return True
