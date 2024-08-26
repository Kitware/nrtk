# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from pathlib import Path
from typing import List

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import nrtk  # noqa: E402

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "nrtk"
copyright = "2023, Kitware, Inc."  # noqa: A001
author = "Kitware, Inc."
release = nrtk.__version__

# -- Support git-lfs on RTD --------------------------------------------------
# https://github.com/InfinniPlatform/InfinniPlatform.readthedocs.org.en/blob/6c13503ca9af83d23faaf2070b6b024046fe23e8/docs/source/conf.py#L18-L31
DOC_SOURCES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(DOC_SOURCES_DIR))
sys.path.insert(0, DOC_SOURCES_DIR)
print("PROJECT_ROOT_DIR", PROJECT_ROOT_DIR)

# If runs on ReadTheDocs environment
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

# Hack for lacking git-lfs support ReadTheDocs
if on_rtd:
    print("Fetching files with git_lfs")
    from git_lfs import fetch

    fetch(PROJECT_ROOT_DIR)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx-prompt",
    "sphinx_copybutton",
]

suppress_warnings = [
    # Suppressing duplicate label warning from autosectionlabel extension.
    # This happens a lot across files that happen to talk about the same
    # topics.
    "autosectionlabel.*",
]

templates_path: List[str] = list()  # ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path: List[str] = list()  # ['_static']
