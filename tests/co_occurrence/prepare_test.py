import os

from penelope.co_occurrence import CoOccurrenceHelper

from . import utils as test_utils

jj = os.path.join


def test_co_occurrence_helper_reset():

    helper: CoOccurrenceHelper = test_utils.create_simple_helper()

    helper.reset()

    assert (helper.data == helper.co_occurrences).all().all()


def test_co_occurrence_groupby():

    helper: CoOccurrenceHelper = test_utils.create_simple_helper()

    helper.reset()

    yearly_co_occurrences = helper.groupby('year').value

    assert yearly_co_occurrences is not None
