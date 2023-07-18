from io import StringIO
from typing import Sequence

import pandas as pd

from penelope.corpus import term_frequency
from penelope.type_alias import SerializableContent
from penelope.utility import registry

from .interface import ContentType, IContentSerializer, SerializeOpts


class SerializerRegistry(registry.Registry[IContentSerializer]):
    @classmethod
    def create(cls, opts: SerializeOpts) -> "IContentSerializer":
        if opts.custom_serializer:
            return opts.custom_serializer()

        if cls.registered(opts.content_type):
            return cls.get(opts.content_type)()

        raise ValueError(f"non-serializable content type: {opts.content_type}")


# pylint: disable=unused-argument, no-member


@SerializerRegistry.register(key=ContentType.TEXT)
class TextContentSerializer(IContentSerializer):
    def serialize(self, *, content: SerializableContent, options: SerializeOpts) -> str:
        return content

    def deserialize(self, *, content: str, options: SerializeOpts) -> SerializableContent:
        return content

    def compute_term_frequency(self, *, content: SerializableContent, options: SerializeOpts) -> dict:
        return {}


@SerializerRegistry.register(key=ContentType.TOKENS)
class TokensContentSerializer(IContentSerializer):
    def serialize(self, *, content: SerializableContent, options: SerializeOpts) -> str:
        return ' '.join(content)

    def deserialize(self, *, content: str, options: SerializeOpts) -> Sequence[str]:
        return content.split(' ')

    def compute_term_frequency(self, *, content: SerializableContent, options: SerializeOpts) -> dict:
        return dict(term_frequency=term_frequency(content))


@SerializerRegistry.register(key=ContentType.TAGGED_FRAME)
class CsvContentSerializer(IContentSerializer):
    def serialize(self, *, content: SerializableContent, options: SerializeOpts) -> str:
        return content.to_csv(sep=options.sep, header=True)

    def deserialize(self, *, content: str, options: SerializeOpts) -> SerializableContent:
        data: pd.DataFrame = pd.read_csv(
            StringIO(content),
            sep=options.sep,
            quoting=options.quoting,
            index_col=options.index_column,
            dtype={
                options.lemma_column: str,
                options.text_column: str,
                options.pos_column: str,
            },
        )
        data.fillna("", inplace=True)
        if any(x not in data.columns for x in options.columns):
            raise ValueError(f"missing columns: {', '.join([x for x in options.columns if x not in data.columns])}")
        if options.lower_lemma:
            data[options.lemma_column] = pd.Series([x.lower() for x in data[options.lemma_column]], dtype=object)
        return data[options.columns]

    def compute_term_frequency(self, *, content: SerializableContent, options: SerializeOpts) -> dict:
        if not options.frequency_column:
            return {}

        return dict(
            term_frequency=term_frequency(content[options.frequency_column]),
            pos_frequency=term_frequency(content[options.pos_column]),
        )
