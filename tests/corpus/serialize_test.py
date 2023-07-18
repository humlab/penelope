from penelope.corpus.serialize import ContentType, SerializeOpts, SerializerRegistry


def test_serializer_registry():
    assert SerializerRegistry.get('text') is not None
    assert SerializerRegistry.get('tokens') is not None
    assert SerializerRegistry.get('tagged_frame') is not None


def test_serialize_opts():
    assert SerializeOpts.create({'content_type': ContentType.TAGGED_FRAME}).content_type == ContentType.TAGGED_FRAME
    assert SerializeOpts().content_type == 'text'
