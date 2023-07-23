from penelope.corpus.serialize import ContentType, SerializeOpts, SerializerRegistry


def test_serializer_registry():
    assert SerializerRegistry.get(ContentType.TEXT) is not None
    assert SerializerRegistry.get(ContentType.TAGGED_FRAME) is not None
    assert SerializerRegistry.get(ContentType.TOKENS) is not None


def test_serialize_opts():
    assert SerializeOpts.create({'content_type': ContentType.TAGGED_FRAME}).content_type == ContentType.TAGGED_FRAME
    assert SerializeOpts().content_type == ContentType.NONE
