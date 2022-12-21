import os
from dataclasses import dataclass, field
from typing import List, Mapping, Optional

from penelope.co_occurrence import ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusType

# pylint: disable=useless-super-delegation, no-member


def ingest(obj: any, data: dict, properties: List[str] = None):
    if isinstance(properties, str):
        properties = [properties]
    if properties is not None:
        properties = [p for p in properties if p in data]
    for key in properties or data:
        if hasattr(obj, key):
            setattr(obj, key, data[key])


@dataclass
class ComputeOptBase:

    corpus_type: CorpusType = CorpusType.Tokenized
    corpus_source: Optional[str] = None
    target_folder: Optional[str] = None
    corpus_tag: Optional[str] = None
    create_subfolder: bool = True
    persist: bool = True

    tagged_corpus_source: str = field(init=True, default=None)

    dry_run: bool = field(init=False, default=False)

    def is_satisfied(self):

        if not self.corpus_source:
            raise ValueError("please specify corpus source")

        if not os.path.isfile(self.corpus_source) and not os.path.isdir(self.corpus_source):
            raise FileNotFoundError(self.corpus_source)

        if not self.corpus_tag:
            raise ValueError("please specify output tag")

        if not self.target_folder:
            raise ValueError("please specify output folder")

        return True

    @property
    def props(self) -> dict:
        options = {
            'corpus_type': int(self.corpus_type),
            'input_filename': self.corpus_source,
            'output_folder': self.target_folder,
            'output_tag': self.corpus_tag,
        }
        return options

    def cli_options(self) -> Mapping[str, str]:
        options = {}
        return options

    def ingest(self, data: dict):
        ingest(self, data)


@dataclass
class CheckpointOptsMixIn:

    enable_checkpoint: bool = field(init=True, default=True)
    force_checkpoint: bool = field(init=True, default=False)

    @property
    def props(self):
        d: dict = super().props
        d.update(enable_checkpoint=self.enable_checkpoint, force_checkpoint=self.force_checkpoint)
        return d

    def is_satisfied(self):
        return super().is_satisfied()

    def cli_options(self) -> Mapping[str, str]:

        options: Mapping[str, str] = {}

        if self.enable_checkpoint:
            options['--enable-checkpoint'] = True

        if self.force_checkpoint:
            options['--force-checkpoint'] = True

        options.update(super().cli_options())

        return options

    def ingest(self, data: dict):
        super().ingest(data)
        ingest(self, data)


@dataclass
class ReaderOptsMixIn:

    text_reader_opts: TextReaderOpts = None

    @property
    def props(self):
        return {**super().props, **dict(text_reader_opts=self.text_reader_opts.props)}

    def is_satisfied(self):
        if len(self.text_reader_opts.filename_fields or []) == 0:
            raise ValueError("please specify at least one filename field")
        return super().is_satisfied()

    def cli_options(self) -> Mapping[str, str]:

        options: Mapping[str, str] = {}

        # if self.text_reader_opts.filename_fields:
        #     options['--filename-field'] = self.text_reader_opts.filename_fields

        options.update(super().cli_options())

        return options

    def ingest(self, data: dict):
        super().ingest(data)
        ingest(self.text_reader_opts, data)


@dataclass
class VectorizeOptsMixIn:

    vectorize_opts: VectorizeOpts = None

    @property
    def props(self) -> dict:
        return {**super().props, **dict(vectorize_opts=self.vectorize_opts.props)}

    def is_satisfied(self):
        return super().is_satisfied()

    def cli_options(self) -> Mapping[str, str]:

        options: dict = {}

        if self.vectorize_opts.lowercase:
            options['--to-lower'] = self.vectorize_opts.lowercase

        if self.vectorize_opts.max_df:
            options['--max-df'] = self.vectorize_opts.max_df

        if self.vectorize_opts.min_df:
            options['--min-df'] = self.vectorize_opts.min_df

        if self.vectorize_opts.stop_words:
            options['--remove-stopwords'] = self.vectorize_opts.stop_words

        # if self.vectorize_opts.already_tokenized:
        #     options['--already-tokenized'] = self.vectorize_opts.already_tokenized

        options.update(super().cli_options())

        return options

    def ingest(self, data: dict):
        super().ingest(data)
        ingest(self.vectorize_opts, data)


@dataclass
class TransformOptsMixIn:

    transform_opts: TokensTransformOpts = None

    @property
    def props(self) -> dict:
        return {**super().props, **dict(transform_opts=self.transform_opts.props)}

    def is_satisfied(self):
        return super().is_satisfied()

    def cli_options(self) -> Mapping[str, str]:

        options: dict = {}

        if self.transform_opts.to_lower:
            options['--to-lower'] = True

        options[f'--{"" if self.transform_opts.keep_symbols else "no-" }keep-symbols'] = True
        options[f'--{"" if self.transform_opts.keep_numerals else "no-" }keep-numerals'] = True

        if self.transform_opts.min_len > 1:
            options['--min-word-length'] = self.transform_opts.min_len

        if (self.transform_opts.max_len or 99) < 99:
            options['--max-word-length'] = self.transform_opts.max_len

        if self.transform_opts.remove_stopwords:
            options['--remove-stopwords'] = self.transform_opts.language

        if self.transform_opts.only_alphabetic:
            options['--only-alphabetic'] = True

        if self.transform_opts.only_any_alphanumeric:
            options['--only-any-alphanumeric'] = True

        options.update(super().cli_options())

        return options

    def ingest(self, data: dict):
        super().ingest(data)


@dataclass
class ExtractTaggedTokensOptsMixIn:

    extract_opts: ExtractTaggedTokensOpts = None
    tf_threshold: Optional[int] = 1
    tf_threshold_mask: Optional[bool] = False

    @property
    def props(self) -> dict:
        return {
            **super().props,
            **dict(extract_opts=self.extract_opts.props),
            **dict(tf_threshold=self.tf_threshold, tf_threshold_mask=self.tf_threshold_mask),
        }

    def is_satisfied(self):
        return super().is_satisfied()

    def cli_options(self) -> Mapping[str, str]:

        options: dict = {}

        if self.extract_opts.phrases and len(self.extract_opts.phrases) > 0:
            options['--phrase'] = self.extract_opts.phrases

        if self.extract_opts.pos_includes:
            options['--pos-includes'] = self.extract_opts.pos_includes

        if self.extract_opts.pos_paddings and self.extract_opts.pos_paddings.strip('|'):
            options['--pos-paddings'] = self.extract_opts.pos_paddings

        if self.extract_opts.pos_excludes:
            options['--pos-excludes'] = self.extract_opts.pos_excludes

        if self.extract_opts.lemmatize:
            options['--lemmatize'] = True

        if self.extract_opts.append_pos:
            options['--append-pos'] = True

        if self.tf_threshold > 1:
            options['--tf-threshold'] = self.tf_threshold

        if self.tf_threshold_mask:
            options['--tf-threshold-mask'] = self.tf_threshold_mask

        options.update(super().cli_options())

        return options

    def ingest(self, data: dict):
        super().ingest(data)
        ingest(self.extract_opts, data)
        ingest(self, data, ['tf_threshold', 'tf_threshold_mask'])


@dataclass
class ContextOptsMixIn:

    context_opts: Optional[ContextOpts] = None

    @property
    def props(self) -> dict:
        return {**super().props, **dict(context_opts=self.context_opts.props)}

    def is_satisfied(self):

        if self.context_opts:

            # if len(self.context_opts.concept or []) == 0:
            #     raise ValueError("please specify at least one concept")

            if self.context_opts.context_width is None:
                raise ValueError("please specify at width of context as max distance from concept")

            # if len(self.context_opts.partition_keys or []) == 0:
            #     raise ValueError("please specify partition key")

            # if len(self.context_opts.partition_keys) > 1:
            #     raise ValueError("only one partition key is allowed (for now)")

        return super().is_satisfied()

    def cli_options(self) -> Mapping[str, str]:

        options = {}

        if self.context_opts:

            options['--context-width'] = self.context_opts.context_width

            if len(self.context_opts.concept or []) > 0:
                options['--concept'] = self.context_opts.concept

            # if len(self.context_opts.partition_keys or []) > 0:
            #     options['--partition-key'] = self.context_opts.partition_keys

            # FIXME This could be an array of concepts?
            if self.context_opts.ignore_concept:
                options['--ignore-concept'] = self.context_opts.ignore_concept

            if self.context_opts.ignore_padding:
                options['--ignore-padding'] = self.context_opts.ignore_padding

        options.update(super().cli_options())
        return options

    def ingest(self, data: dict):
        super().ingest(data)
        ingest(self.context_opts, data)


@dataclass
class ComputeOpts(
    ContextOptsMixIn,
    ExtractTaggedTokensOptsMixIn,
    TransformOptsMixIn,
    VectorizeOptsMixIn,
    ReaderOptsMixIn,
    CheckpointOptsMixIn,
    ComputeOptBase,
):
    def is_satisfied(self):
        return super().is_satisfied()

    @property
    def props(self) -> dict:
        return super().props

    def cli_options(self) -> Mapping[str, str]:
        return super().cli_options()

    def ingest(self, data: dict):
        super().ingest(data)
