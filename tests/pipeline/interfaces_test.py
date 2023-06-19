from penelope.pipeline.interfaces import PipelinePayload


def test_payload_update_path():
    assert PipelinePayload.update_path('/gorilla', '/chimpans/apa/test.txt', 'replace') == '/gorilla/test.txt'
    assert PipelinePayload.update_path('/gorilla', '/chimpans/apa', 'replace') == '/gorilla/apa'
