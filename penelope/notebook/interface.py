import os
from dataclasses import dataclass, field
from typing import List, Mapping

from loguru import logger
from penelope.co_occurrence import ContextOpts, folder_and_tag_to_filename
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts
from penelope.corpus.dtm import VectorizeOpts
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

    dry_run: bool = field(init=False, default=False)

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

    @property
    def props(self) -> dict:
        options = {
            'corpus_type': int(self.corpus_type),
            'input_filename': self.corpus_filename,
            'output_folder': self.target_folder,
            'output_tag': self.corpus_tag,
            'tokens_transform_opts': self.tokens_transform_opts.props,
            'reader_opts': self.text_reader_opts.props,
            'extract_tokens_opts': self.extract_tagged_tokens_opts.props
            if self.extract_tagged_tokens_opts is not None
            else {},
            'vectorize_opt': self.vectorize_opts.props,
            'count_threshold': self.count_threshold,
            'context_opts': self.context_opts.props,
            'partition_keys': None if self.partition_keys is None else list(self.partition_keys),
        }
        return options

    def command_line_options(self) -> Mapping[str, str]:

        options = {}

        if self.context_opts:

            options['--context-width'] = self.context_opts.context_width
            if self.context_opts.ignore_concept:
                options['--no-concept'] = True

            if len(self.context_opts.concept or []) > 0:
                options['--concept'] = self.context_opts.concept

        options['--count-threshold'] = self.count_threshold

        if self.extract_tagged_tokens_opts.phrases and len(self.extract_tagged_tokens_opts.phrases) > 0:
            options['--phrase'] = self.extract_tagged_tokens_opts.phrases

        if self.extract_tagged_tokens_opts.pos_includes:
            options['--pos-includes'] = self.extract_tagged_tokens_opts.pos_includes

        if self.extract_tagged_tokens_opts.pos_paddings:
            options['--pos-paddings'] = self.extract_tagged_tokens_opts.pos_paddings

        if self.extract_tagged_tokens_opts.pos_excludes:
            options['--pos-excludes'] = self.extract_tagged_tokens_opts.pos_excludes

        if len(self.partition_keys or []) > 0:
            options['--partition-key'] = self.partition_keys

        if self.extract_tagged_tokens_opts.lemmatize:
            options['--lemmatize'] = True

        if self.tokens_transform_opts.to_lower:
            options['--to-lowercase'] = True

        if self.extract_tagged_tokens_opts.append_pos:
            options['--append-pos'] = True

        options[f'--{"" if self.tokens_transform_opts.keep_symbols else "no" }-keep-symbols'] = True
        options[f'--{"" if self.tokens_transform_opts.keep_numerals else "no" }-keep-numerals'] = True
        options[f'--{"" if self.tokens_transform_opts.to_lower else "no" }-to-lowercase'] = True

        if self.tokens_transform_opts.min_len > 1:
            options['--min-word-length'] = self.tokens_transform_opts.min_len

        if (self.tokens_transform_opts.max_len or 99) < 99:
            options['--max-word-length'] = self.tokens_transform_opts.max_len

        if self.tokens_transform_opts.remove_stopwords:
            options['--remove_stopwords'] = self.tokens_transform_opts.language

        if self.tokens_transform_opts.only_alphabetic:
            options['--only-alphabetic'] = True

        if self.tokens_transform_opts.only_any_alphanumeric:
            options['--only-any-alphanumeric'] = True

        if self.force:
            options['--force'] = True

        return options

    def command_line(self, script: str) -> str:

        options: List[str] = []

        for key, value in self.command_line_options().items():
            if isinstance(value, bool):
                options.append(key)
            elif isinstance(value, (str,)):
                options.append(f"{key} \"{value}\"")
            elif isinstance(
                value,
                (
                    str,
                    int,
                ),
            ):
                options.append(f"{key} {value}")
            elif isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                options.extend([f"{key} \"{v}\"" for v in value])
            else:
                logger.warning(f"skipped option {key} {value}")

        config_filename: str = "doit.yml"
        target_filename: str = folder_and_tag_to_filename(folder=self.target_folder, tag=self.corpus_tag)
        command: str = f"{script} {' '.join(options)} {config_filename} {self.corpus_filename} {target_filename}"

        return command
