from os.path import isfile
from typing import Any, Protocol

import pandas as pd

from penelope.type_alias import TaggedFrame, TaggedFrameStore
from penelope.utility import registry, replace_folder_and_extension, zipfile_or_filename

from .interface import ContentType, SerializeOpts, Serializer


class ILoader(Protocol):
    def __call__(
        self,
        zip_or_filename: str,
        opts: SerializeOpts,
        filenames: list[str],
        ordered: bool = False,
    ) -> Any:
        ...


class LoaderRegistry(registry.Registry[ILoader]):
    @classmethod
    def get_loader(cls, opts: SerializeOpts) -> ILoader:
        if opts.content_type == ContentType.TAGGED_FRAME:
            if opts.feather_folder:
                return load_feathered_tagged_frame
            return load_tagged_frame

        raise NotImplementedError("Loader so far only implemented for TAGGED FRAME")


@LoaderRegistry.register(key="tagged_frame")
@zipfile_or_filename(mode='r')
def load_tagged_frame(
    *, zip_or_filename: TaggedFrameStore, filename: str, opts: SerializeOpts, serializer: Serializer
) -> TaggedFrame:
    tagged_frame: TaggedFrame = serializer.deserialize(
        content=zip_or_filename.read(filename).decode(encoding='utf-8'), options=opts
    )
    if opts.lower_lemma:
        tagged_frame[opts.lemma_column] = pd.Series([x.lower() for x in tagged_frame[opts.lemma_column]], dtype=object)
    return tagged_frame


@LoaderRegistry.register(key="tagged_frame_feather")
def load_feathered_tagged_frame(
    *, zip_or_filename: TaggedFrameStore, filename: str, opts: SerializeOpts, serializer: Serializer
) -> pd.DataFrame:
    feather_filename: str = replace_folder_and_extension(filename, opts.feather_folder, '.feather')
    if isfile(feather_filename):
        tagged_frame: pd.DataFrame = pd.read_feather(feather_filename)
        if opts.lower_lemma:
            if len(tagged_frame) > 0:
                tagged_frame[opts.lemma_column] = tagged_frame[opts.lemma_column].str.lower()
    else:
        tagged_frame = load_tagged_frame(
            zip_or_filename=zip_or_filename, filename=filename, opts=opts, serializer=serializer
        )
        tagged_frame.reset_index(drop=True).to_feather(feather_filename, compression="lz4")
    return tagged_frame
