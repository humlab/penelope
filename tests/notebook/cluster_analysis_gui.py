
from unittest import mock

import pandas as pd
import penelope.notebook.cluster_analysis_gui as cluster_analysis_gui
from penelope.corpus import VectorizedCorpus


def test_display_gui():

    corpus = mock.Mock(spec=VectorizedCorpus)
    df_gof = mock.Mock(spec=pd.DataFrame)
    container = cluster_analysis_gui.create_gui(corpus, df_gof)

    assert container is not None
