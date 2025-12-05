"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import nrtk

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "nrtk"
copyright = "2023, Kitware, Inc."  # noqa: A001
author = "Kitware, Inc."
release = nrtk.__version__

# -- Version switcher configuration ------------------------------------------
# Determine version for switcher matching based on ReadTheDocs or git branch
rtd_version = os.environ.get("READTHEDOCS_VERSION")

if rtd_version:
    # Building on ReadTheDocs - use their version identifier
    # "latest" -> main branch builds
    # "stable" -> latest release
    # "v0.26.0" -> specific tagged version
    switcher_version = rtd_version
    if rtd_version.startswith("v"):
        # Remove 'v' prefix for version matching (e.g., "v0.26.0" -> "0.26.0")
        switcher_version = rtd_version[1:]
else:
    try:
        git_path = shutil.which("git")
        if not git_path:
            raise FileNotFoundError("git executable not found in PATH")
        result = subprocess.run(  # noqa: S603
            [git_path, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        branch = result.stdout.strip()

        # Map branches to switcher versions
        # main branch -> "latest", other branches -> package release version
        switcher_version = "latest" if branch == "main" else release
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git not available or not in a repo - use release version
        switcher_version = release


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.imgconverter",
    "sphinx-prompt",
    "sphinx_copybutton",
    "sphinx_click",
    "sphinx_design",
    "sphinxcontrib.jquery",
    "sphinx_datatables",
    "myst_nb",
]

suppress_warnings = [
    # Suppressing duplicate label warning from autosectionlabel extension.
    # This happens a lot across files that happen to talk about the same
    # topics.
    "autosectionlabel.*",
]

# Autosummary templates reference link:
# https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/tree/master
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

# -- PyData Sphinx Theme configuration ---------------------------------------
html_theme_options = {
    "logo": {
        "image_light": "figures/nrtk-wordmark.png",
        "image_dark": "figures/nrtk-wordmark.png",
        "alt_text": "NRTK Documentation",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["search-button", "theme-switcher", "version-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": switcher_version,
    },
    "show_nav_level": 1,  # Show only top-level navigation initially
    "navigation_depth": 3,  # Navigation hierarchy depth in sidebar
    "collapse_navigation": True,  # Enable collapse/expand functionality
    "show_toc_level": 1,  # Table of contents depth on page
    "navigation_with_keys": True,
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Kitware/nrtk",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nrtk/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
        {
            "name": "conda-forge",
            "url": "https://github.com/conda-forge/nrtk-feedstock",
            "icon": "fa-solid fa-box-open",
            "type": "fontawesome",
        },
    ],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": False,
}

# Configure sidebar templates
# NOTE: Intentionally NOT including "search-field.html" to avoid duplicate search UI
# The search-button in navbar_end provides search functionality
# Use sidebar-nav-bs.html for section-isolated navigation (like pandas docs)
html_sidebars = {
    "index": [],  # No sidebar on main landing page
    "getting_started/**": ["sidebar-nav-bs"],
    "tutorials/**": ["sidebar-nav-bs"],
    "how_to_guides/**": ["sidebar-nav-bs"],
    "explanations/**": ["sidebar-nav-bs"],
    "reference/**": ["sidebar-nav-bs"],
    "interoperability/**": ["sidebar-nav-bs"],
    "development/**": ["sidebar-nav-bs"],
    "release_notes/**": ["sidebar-nav-bs"],
}

html_static_path = ["_static"]
html_css_files = [
    "https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css",
    "css/custom.css",
]
html_js_files = [
    "https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js",
    "https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js",
    "https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js",
    "javascript/custom.js",
]

# -- MyST-NB settings---------------------------------------------------------
nb_execution_mode = "off"

# -- LaTeX engine ------------------------------------------------------------
latex_engine = "lualatex"

# -- Datatables --------------------------------------------------------------
# set the version to use for DataTables plugin
datatables_version = "2.3.0"

# name of the class to use for tables to enable DataTables
datatables_class = "sphinx-datatable"

datatables_options = {
    "pageLength": -1,
    "paging": False,
    "scrollX": False,
    "layout": {
        "topStart": "search",
        "topEnd": "buttons",
        "bottomStart": None,
        "bottomEnd": None,
    },
}
