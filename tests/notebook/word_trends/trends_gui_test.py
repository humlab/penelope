import unittest.mock as mock

import ipywidgets

from penelope.corpus import VectorizedCorpus
from penelope.notebook.word_trends import ITrendDisplayer, TrendsGUI, TrendsService


def mocked_displayer_ctor(**_):
    m = mock.MagicMock(ITrendDisplayer)
    m.name = "apa"
    m.titles = "apa"
    return m


def test_TrendsGUI_setup():
    gui = TrendsGUI().setup(displayers=[mocked_displayer_ctor])
    assert len(gui._displayers) == 1  # pylint: disable=protected-access


def test_TrendsGUI_layout():
    w = TrendsGUI().setup(displayers=[mocked_displayer_ctor]).layout()
    assert isinstance(w, ipywidgets.CoreWidget)


def test_TrendsGUI_display():
    corpus = mock.Mock(spec=VectorizedCorpus)
    trends_service = mock.MagicMock(spec=TrendsService, corpus=corpus, category_column="apa")
    gui = TrendsGUI().setup(displayers=[mocked_displayer_ctor])
    gui.trends_service = trends_service
    gui.display(trends_service=trends_service)


import os
import uuid
from os.path import dirname, isfile
from unittest.mock import Mock, patch

from penelope import pipeline
from penelope.corpus import VectorizedCorpus
from penelope.notebook import pick_file_gui as pfg
from penelope.notebook import word_trends as wt
from penelope.notebook.word_trends import main_gui
from penelope.utility import PivotKeys
from penelope.workflows import interface

# from ..utils import create_abc_corpus


def monkey_patch(*_, **__):
    ...


def find_corpus_config(*_, **__) -> pipeline.CorpusConfig:
    corpus_config: pipeline.CorpusConfig = pipeline.CorpusConfig.load('./tests/test_data/SSI.yml')
    return corpus_config


# def test_corpus_loaded_callback():
#     corpus = create_abc_corpus(
#         dtm=[
#             [2, 1, 4, 1],
#             [2, 2, 3, 0],
#             [2, 3, 2, 0],
#             [2, 4, 1, 1],
#             [2, 0, 1, 1],
#         ],
#         document_years=[2013, 2013, 2014, 2014, 2014],
#     )
#     main_gui.loaded_callback(corpus, folder='./tests/output')


# @patch('penelope.workflows.vectorize.dtm.compute', monkey_patch)
# def test_corpus_compute_callback():
#     main_gui.compute_callback(args=Mock(spec=interface.ComputeOpts), corpus_config=Mock(spec=pipeline.CorpusConfig))


# @patch('penelope.pipeline.CorpusConfig.find', find_corpus_config)
# def test_create_gui():
#     corpus_folder = f'./tests/output/{uuid.uuid1()}'
#     data_folder = f'./tests/output/{uuid.uuid1()}'
#     os.makedirs(corpus_folder)
#     os.makedirs(data_folder)
#     gui = main_gui.create_advanced_dtm_gui(corpus_folder=corpus_folder, data_folder=data_folder, corpus_config=None)
#     assert gui is not None


# def display_callback(filename: str, sender: pfg.PickFileGUI):
#     print(f"filename: {filename}")
#     if not VectorizedCorpus.is_dump(filename):
#         raise ValueError(f"Expected a DTM file, got {filename or 'None'}")

#     corpus = VectorizedCorpus.load(filename=filename)
#     folder = dirname(filename)
#     trends_service: wt.TrendsService = wt.TrendsService(corpus=corpus, n_top=25000)
#     pivot_keys: PivotKeys = PivotKeys.load(folder) if isfile(folder) else None
#     gui: wt.TrendsGUI = wt.TrendsGUI(pivot_key_specs=pivot_keys).setup(displayers=wt.DEFAULT_WORD_TREND_DISPLAYERS)
#     sender.add(gui.layout())
#     gui.display(trends_service=trends_service)


# def test_create_simple_gui():
#     gui: pfg.PickFileGUI = pfg.PickFileGUI(
#         folder="/data/inidun/courier/dtm/v0.2.0",
#         pattern='*_vector_data.npz',
#         picked_callback=display_callback,
#         kind='picker',
#     ).setup()
#     gui.layout()
