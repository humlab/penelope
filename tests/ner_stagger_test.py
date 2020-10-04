import os
import pytest # pylint: disable=unused-import

import penelope.ner.stagger_wrapper as stagger

STAGGER_HOME = os.environ.get('STAGGER_HOME', './lib/stagger')

def stagger_is_found():

    return os.path.isfile(os.path.join(STAGGER_HOME, "stagger.jar"))

@pytest.mark.skipif(not stagger_is_found(), reason="requires stagger.jar not found")
def test_create_wrapper_by_env():

    os.environ['STAGGER_HOME'] = STAGGER_HOME

    wrapper = stagger.StaggerWrapper()

    assert wrapper is not None
    assert wrapper.stagger_jar_path == os.path.join(STAGGER_HOME, "stagger.jar")
    assert wrapper.stagger_model_path == os.path.join(STAGGER_HOME, "swedish.bin")

@pytest.mark.skipif(not stagger_is_found(), reason="requires stagger.jar not found")
def test_create_wrapper_by_args():

    os.environ['STAGGER_HOME'] = STAGGER_HOME
    stagger_jar_path = os.path.join(STAGGER_HOME, "stagger.jar")
    stagger_model_path = os.path.join(STAGGER_HOME, "swedish.bin")

    wrapper = stagger.StaggerWrapper(stagger_jar_path, stagger_model_path)

    assert wrapper is not None
    assert wrapper.stagger_jar_path == os.path.join(STAGGER_HOME, "stagger.jar")
    assert wrapper.stagger_model_path == os.path.join(STAGGER_HOME, "swedish.bin")

@pytest.mark.skipif(not stagger_is_found(), reason="requires stagger.jar not found")
def test_build_command():

    os.environ['STAGGER_HOME'] = STAGGER_HOME
    stagger_jar_path = os.path.join(STAGGER_HOME, "stagger.jar")
    stagger_model_path = os.path.join(STAGGER_HOME, "swedish.bin")

    wrapper = stagger.StaggerWrapper(stagger_jar_path, stagger_model_path)

    command = wrapper.build_command("*.txt", "16G")

    assert isinstance(command, str)
