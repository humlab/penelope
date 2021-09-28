from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Container, Optional, Union

from penelope.corpus import TokensTransformer, TokensTransformOpts, VectorizeOpts
from penelope.corpus.interfaces import ITokenizedCorpus
from penelope.utility import PropertyValueMaskingOpts, deprecated

from . import tasks

if TYPE_CHECKING:
    from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts, TextTransformOpts

    from . import pipelines
    from .checkpoint import CheckpointOpts
# pylint: disable=too-many-public-methods


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
        """ _ => DATAFRAME """
        return self.add(
            tasks.LoadTaggedCSV(filename=filename, checkpoint_opts=checkpoint_opts, extra_reader_opts=extra_reader_opts)
        )

    def load_tagged_xml(
        self: pipelines.CorpusPipeline, filename: str, options: TextReaderOpts
    ) -> pipelines.CorpusPipeline:
        """ SparvXML => DATAFRAME """
        return self.add(tasks.LoadTaggedXML(filename=filename, reader_opts=options))

    def checkpoint(
        self: pipelines.CorpusPipeline,
        filename: str,
        checkpoint_opts: CheckpointOpts = None,
        force_checkpoint: bool = False,
    ) -> pipelines.CorpusPipeline:
        """ [DATAFRAME,TEXT,TOKENS] => [CHECKPOINT] => PASSTHROUGH """
        return self.add(
            tasks.Checkpoint(filename=filename, checkpoint_opts=checkpoint_opts, force_checkpoint=force_checkpoint)
        )

    def checkpoint_feather(
        self: pipelines.CorpusPipeline,
        folder: str,
        force: bool = False,
    ) -> pipelines.CorpusPipeline:
        """ [DATAFRAME] => [CHECKPOINT] => PASSTHROUGH """
        return self.add(tasks.CheckpointFeather(folder=folder, force=force))

    def tokens_to_text(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        """ [TOKEN] => TEXT """
        return self.add(tasks.TokensToText())

    def text_to_tokens(
        self,
        *,
        text_transform_opts: TextTransformOpts,
        transform_opts: TokensTransformOpts = None,
        transformer: TokensTransformer = None,
    ) -> pipelines.CorpusPipeline:
        """ TOKEN => TOKENS """
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
        """ TOKEN => TOKENS """
        if transform_opts or transformer:
            return self.add(tasks.TokensTransform(transform_opts=transform_opts, transformer=transformer))
        return self

    def to_dtm(self: pipelines.CorpusPipeline, vectorize_opts: VectorizeOpts = None) -> pipelines.CorpusPipeline:
        """ (filename, TEXT => DTM) """
        return self.add(tasks.TextToDTM(vectorize_opts=vectorize_opts or VectorizeOpts()))

    def to_content(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.ToContent())

    def tqdm(self: pipelines.CorpusPipeline, desc: str = None) -> pipelines.CorpusPipeline:
        return self.add(tasks.Tqdm(desc=desc))

    def passthrough(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.Passthrough())

    def to_document_content_tuple(self: pipelines.CorpusPipeline) -> pipelines.CorpusPipeline:
        return self.add(tasks.ToDocumentContentTuple())

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
    ) -> pipelines.CorpusPipeline:

        token_type: tasks.Vocabulary.TokenType = (
            tasks.Vocabulary.TokenType.Lemma if lemmatize else tasks.Vocabulary.TokenType.Text
        )
        return self.add(
            tasks.Vocabulary(
                token_type=token_type,
                progress=progress,
                tf_threshold=tf_threshold,
                tf_keeps=tf_keeps,
                close=close,
            )
        )

    def tagged_frame_to_tokens(
        self: pipelines.CorpusPipeline,
        *,
        extract_opts: Optional[ExtractTaggedTokensOpts],
        transform_opts: Optional[TokensTransformOpts],
        filter_opts: Optional[PropertyValueMaskingOpts],
    ) -> pipelines.CorpusPipeline:
        return self.add(
            tasks.TaggedFrameToTokens(
                extract_opts=extract_opts,
                filter_opts=filter_opts,
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

    # def filter_tagged_frame(
    #     self: pipelines.CorpusPipeline,
    #     extract_opts: ExtractTaggedTokensOpts,
    #     filter_opts: PropertyValueMaskingOpts,
    # ) -> pipelines.CorpusPipeline:
    #     return self.add(
    #         tasks.FilterTaggedFrame(
    #             extract_opts=extract_opts,
    #             filter_opts=filter_opts,
    #         )
    #     )
