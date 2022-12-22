from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Container, Optional, Union

from penelope.corpus import ITokenizedCorpus, TokensTransformer, TokensTransformOpts
from penelope.utility import PoS_Tag_Scheme, deprecated

from . import tagged_frame, tasks

if TYPE_CHECKING:
    from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts, TextTransformOpts

    from . import pipelines
    from .checkpoint import CheckpointOpts

# pylint: disable=too-many-public-methods, no-member


class PipelineShortcutMixIn:
    """Shortcuts for specific tasks that can be injected to derived pipelines"""

    def load_text(
        self: pipelines.CorpusPipeline,
        *,
        reader_opts: TextReaderOpts = None,
        transform_opts: TextTransformOpts = None,
        source=None,
    ) -> pipelines.CorpusPipeline:
        return self.add(tasks.LoadText(source=source, reader_opts=reader_opts, transform_opts=transform_opts))

    def load_corpus(self, corpus: ITokenizedCorpus) -> pipelines.CorpusPipeline:
        return self.add(tasks.LoadTokenizedCorpus(corpus=corpus))

    @deprecated
    def write_feather(self, folder: str) -> pipelines.CorpusPipeline:
        return self.add(tasks.WriteFeather(folder=folder))

    @deprecated
    def read_feather(self, folder: str) -> pipelines.CorpusPipeline:
        return self.add(tasks.ReadFeather(folder=folder))

    def save_tagged_frame(
        self: pipelines.CorpusPipeline, filename: str, checkpoint_opts: CheckpointOpts
    ) -> pipelines.CorpusPipeline:
        return self.add(tasks.SaveTaggedCSV(filename=filename, checkpoint_opts=checkpoint_opts))

    def load_tagged_frame(
        self: pipelines.CorpusPipeline,
        filename: str,
        checkpoint_opts: CheckpointOpts,
        extra_reader_opts: TextReaderOpts = None,
    ) -> pipelines.CorpusPipeline:
        """_ => DATAFRAME"""
        return self.add(
            tasks.LoadTaggedCSV(filename=filename, checkpoint_opts=checkpoint_opts, extra_reader_opts=extra_reader_opts)
        )

    def load_id_tagged_frame(
        self: pipelines.CorpusPipeline,
        folder: str,
        file_pattern: str,
        id_to_token: bool = False,
    ) -> pipelines.CorpusPipeline:
        """_ => DATAFRAME"""
        return self.add(
            tagged_frame.LoadIdTaggedFrame(corpus_source=folder, file_pattern=file_pattern, id_to_token=id_to_token)
        )

    def to_id_tagged_frame(
        self: pipelines.CorpusPipeline,
        ingest_vocab_type: str = tagged_frame.IngestVocabType.Incremental,
    ) -> pipelines.CorpusPipeline:
        """_ => DATAFRAME"""
        return self.add(tagged_frame.ToIdTaggedFrame(ingest_vocab_type=ingest_vocab_type))

    def store_id_tagged_frame(
        self: pipelines.CorpusPipeline,
        folder: str,
    ) -> pipelines.CorpusPipeline:
        """_ => DATAFRAME"""
        return self.add(tagged_frame.StoreIdTaggedFrame(folder=folder))

    def load_tagged_xml(
        self: pipelines.CorpusPipeline, filename: str, options: TextReaderOpts
    ) -> pipelines.CorpusPipeline:
        """SparvXML => DATAFRAME"""
        return self.add(tasks.LoadTaggedXML(filename=filename, reader_opts=options))

    def checkpoint(
        self: pipelines.CorpusPipeline,
        filename: str,
        checkpoint_opts: CheckpointOpts = None,
        force_checkpoint: bool = False,
    ) -> pipelines.CorpusPipeline:
        """[DATAFRAME,TEXT,TOKENS] => [CHECKPOINT] => PASSTHROUGH"""
        return self.add(
            tasks.Checkpoint(filename=filename, checkpoint_opts=checkpoint_opts, force_checkpoint=force_checkpoint)
        )

    def checkpoint_feather(
        self: pipelines.CorpusPipeline,
        folder: str,
        force: bool = False,
    ) -> pipelines.CorpusPipeline:
        """[DATAFRAME] => [CHECKPOINT] => PASSTHROUGH"""
        return self.add(tasks.CheckpointFeather(folder=folder, force=force))

    def tokens_to_text(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        """[TOKEN] => TEXT"""
        return self.add(tasks.TokensToText())

    def text_to_tokens(
        self,
        *,
        text_transform_opts: TextTransformOpts,
        transform_opts: TokensTransformOpts = None,
        transformer: TokensTransformer = None,
    ) -> pipelines.CorpusPipeline:
        """TOKEN => TOKENS"""
        return self.add(
            tasks.TextToTokens(
                text_transform_opts=text_transform_opts,
                transform_opts=transform_opts,
                transformer=transformer,
            )
        )

    def tokens_transform(
        self, *, transform_opts: TokensTransformOpts, transformer: TokensTransformer = None
    ) -> pipelines.CorpusPipeline:
        """TOKEN => TOKENS"""
        if transform_opts or transformer:
            return self.add(tasks.TokensTransform(transform_opts=transform_opts, transformer=transformer))
        return self

    def to_content(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.ToContent())

    def tqdm(self: pipelines.CorpusPipeline, desc: str = None) -> pipelines.CorpusPipeline:
        return self.add(tasks.Tqdm(desc=desc))

    def passthrough(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.Passthrough())

    def project(self: pipelines.CorpusPipeline, project: Callable[[Any], Any]) -> pipelines.CorpusPipeline:
        return self.add(tasks.Project(project=project))

    def vocabulary(
        self: pipelines.CorpusPipeline,
        *,
        lemmatize: bool,
        progress: bool = False,
        tf_threshold: int = None,
        tf_keeps: Container[Union[int, str]] = None,
        close: bool = True,
        to_lower: bool = True,
    ) -> pipelines.CorpusPipeline:

        return self.add(
            tasks.Vocabulary(
                token_type=self.encode_token_type(lemmatize, to_lower),
                progress=progress,
                tf_threshold=tf_threshold,
                tf_keeps=tf_keeps,
                close=close,
            )
        )

    def encode_token_type(self, lemmatize, to_lower):

        if lemmatize:
            return tasks.Vocabulary.TokenType.Lemma

        if to_lower:
            return tasks.Vocabulary.TokenType.LowerText

        return tasks.Vocabulary.TokenType.Text

    def filter_tagged_frame(
        self: pipelines.CorpusPipeline,
        extract_opts: ExtractTaggedTokensOpts,
        pos_schema: PoS_Tag_Scheme = None,
        transform_opts: TokensTransformOpts = None,
        normalize_column_names: bool = False,
    ) -> pipelines.CorpusPipeline:

        if (extract_opts is None or extract_opts.of_no_effect) and (
            transform_opts is None or transform_opts.of_no_effect
        ):
            return self

        return self.add(
            tasks.FilterTaggedFrame(
                extract_opts=extract_opts,
                pos_schema=pos_schema,
                transform_opts=transform_opts,
                normalize_column_names=normalize_column_names,
            )
        )

    def tagged_frame_to_tokens(
        self: pipelines.CorpusPipeline,
        *,
        extract_opts: Optional[ExtractTaggedTokensOpts],
        transform_opts: Optional[TokensTransformOpts],
    ) -> pipelines.CorpusPipeline:
        return self.add(
            tasks.TaggedFrameToTokens(
                extract_opts=extract_opts,
                transform_opts=transform_opts,
            )
        )

    def tap_stream(self: pipelines.CorpusPipeline, target: str, tag: str) -> pipelines.CorpusPipeline:
        """Taps the stream into a debug zink."""
        return self.add(tasks.TapStream(target=target, tag=tag, enabled=True))

    # def assert_payload_content(
    #     self: pipelines.CorpusPipeline,
    #     expected_values: Iterable[Any],
    #     comparer=Callable[[Any, Any], bool],
    #     accept_fewer_expected_values: bool = False,
    # ) -> pipelines.CorpusPipeline:
    #     return self.add(
    #         tasks.AssertPayloadContent(
    #             expected_values=expected_values,
    #             comparer=comparer,
    #             accept_fewer_expected_values=accept_fewer_expected_values,
    #         )
    #     )

    def assert_on_payload(
        self: pipelines.CorpusPipeline, payload_test: Callable[[Any], bool], *payload_test_args: Any
    ) -> pipelines.CorpusPipeline:
        return self.add(tasks.AssertOnPayload(payload_test=payload_test, *payload_test_args))

    def assert_on_exit(
        self: pipelines.CorpusPipeline, exit_test: Callable[[Any], bool], *exit_test_args: Any
    ) -> pipelines.CorpusPipeline:
        return self.add(tasks.AssertOnExit(exit_test=exit_test, *exit_test_args))

    def take(
        self: pipelines.CorpusPipeline,
        *,
        n_count: int,
    ) -> pipelines.CorpusPipeline:
        return self.add(tasks.Take(n_count=n_count))
