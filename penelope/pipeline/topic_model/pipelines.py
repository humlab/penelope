from __future__ import annotations

from .. import config, pipelines


def from_grouped_feather_id_to_tagged_frame_pipeline(
    *,
    corpus_config: config.CorpusConfig,
    corpus_source: str = None,
    **_,
):
    """Loads a tagged data frame"""

    corpus_source: str = corpus_source or corpus_config.pipeline_payload.source

    p: pipelines.CorpusPipeline = pipelines.CorpusPipeline(config=corpus_config).load_grouped_id_tagged_frame(
        folder=corpus_source,
        to_tagged_frame=True,
        file_pattern='**/prot-*.feather',
    )

    return p
