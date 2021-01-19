import penelope.corpus.dtm as dtm
from penelope.co_occurrence.interface import ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TaggedTokensFilterOpts, TokensTransformOpts
from penelope.pipeline.tasks import WildcardTask

from .pipeline import CorpusPipelineBase
from .spacy.tasks_mixin import SpacyPipelineShortcutMixIn
from .tasks_mixin import PipelineShortcutMixIn


class CorpusPipeline(
    PipelineShortcutMixIn,
    SpacyPipelineShortcutMixIn,
    CorpusPipelineBase["CorpusPipeline"],
):
    def __add__(self, other: "CorpusPipeline") -> "CorpusPipeline":
        if len(other.tasks) > 0:
            has_wildcard: bool = isinstance(other.tasks[0], WildcardTask)
            self.add(other.tasks[1 if has_wildcard else 0 :])
            if self.payload is not None:
                self.payload.extend(other.payload)
        return self


AnyPipeline = CorpusPipeline


def wildcard() -> CorpusPipeline:
    p: CorpusPipeline = CorpusPipeline(config=None)
    return p


def wildcard_to_DTM_pipeline(
    tokens_transform_opts: TokensTransformOpts = None,
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = None,
    tagged_tokens_filter_opts: TaggedTokensFilterOpts = None,
    vectorize_opts: dtm.VectorizeOpts = None,
):
    try:
        p: CorpusPipeline = (
            wildcard()
            .tagged_frame_to_tokens(
                extract_opts=extract_tagged_tokens_opts,
                filter_opts=tagged_tokens_filter_opts,
            )
            .tokens_transform(tokens_transform_opts=tokens_transform_opts)
            .to_document_content_tuple()
            .tqdm()
            .to_dtm(vectorize_opts=vectorize_opts)
        )
        return p
    except Exception as ex:
        raise ex


def wildcard_to_co_occurrence_pipeline(
    tokens_transform_opts: TokensTransformOpts = None,
    extract_tagged_tokens_opts: ExtractTaggedTokensOpts = None,
    tagged_tokens_filter_opts: TaggedTokensFilterOpts = None,
    context_opts: ContextOpts = None,
    global_threshold_count: int = None,
    partition_column: str = 'year',
):
    try:
        pipeline: CorpusPipeline = (
            wildcard()
            .tagged_frame_to_tokens(
                extract_opts=extract_tagged_tokens_opts,
                filter_opts=tagged_tokens_filter_opts,
            )
            .tokens_transform(tokens_transform_opts=tokens_transform_opts)
            .vocabulary()
            .to_document_content_tuple()
            .to_co_occurrence(
                context_opts=context_opts,
                global_threshold_count=global_threshold_count,
                partition_column=partition_column,
            )
            .tqdm()
        )

        return pipeline

    except Exception as ex:
        raise ex
