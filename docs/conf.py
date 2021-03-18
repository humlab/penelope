# Configuration file for the Sphinx documentation builder.
# flake8: noqa
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(1500)

import penelope

# -- Project information -----------------------------------------------------

#version = penelope.__version__
version = '2021.03.1'
project = 'humlab-penelope'
author = 'Roger Mähler'
release = version

# -- General configuration ---------------------------------------------------
needs_sphinx = "3.0"

extensions = [
    'recommonmark',
    'sphinx.ext.autodoc',
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    'sphinx.ext.ifconfig',
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

#from recommonmark.parser import CommonMarkParser

# -- Configurations for plugins ------------
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
todo_include_todos = True

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
