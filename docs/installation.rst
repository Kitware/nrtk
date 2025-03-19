Installation
============

There are multiple ways to obtain the nrtk package.
The simplest is to install via the :command:`pip` command.
Alternatively, you can install via :command:`conda-forge` command.
For local development, you can use `Poetry`_.

NRTK installation has been tested on Unix and Linux systems.

.. note::
    To install with OpenCV, see instructions `below <#installing-with-opencv>`_.

.. _installation: Poetry-installation_
.. _usage: Poetry-usage_


From :command:`pip`
-------------------

.. prompt:: bash

    pip install nrtk

From :command:`conda-forge`
---------------------------

.. prompt:: bash

    conda install -c conda-forge nrtk


From Source
-----------
The following assumes `Poetry`_ (`installation`_ and `usage`_) is already installed.

`Poetry`_ is used for development of NRTK. Unlike the previous options, `Poetry`_ will not only allows developers to
install any extras they need, but also install developmental dependencies like ``pytest`` and NRTK's linting tools.

Quick Start
^^^^^^^^^^^

.. prompt:: bash

    cd /where/things/should/go/
    git clone https://github.com/kitware/nrtk.git ./
    poetry install


Installing Developer Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following installs both core and development dependencies as
specified in the :file:`pyproject.toml` file, with versions specified
(including for transitive dependencies) in the :file:`poetry.lock` file:

.. prompt:: bash

    poetry install --sync --with linting,tests,docs

Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^
The documentation for nrtk is maintained as a collection of
`reStructuredText`_ documents in the :file:`docs/` folder of the project.
The :program:`Sphinx` documentation tool can process this documentation
into a variety of formats, the most common of which is HTML.

Within the :file:`docs/` directory is a Unix :file:`Makefile` (for Windows
systems, a :file:`make.bat` file with similar capabilities exists).
This :file:`Makefile` takes care of the work required to run :program:`Sphinx`
to convert the raw documentation to an attractive output format.
For example, calling the command below will generate
HTML format documentation rooted at :file:`docs/_build/html/index.html`.

.. prompt:: bash

    poetry run make html

Calling the command ``make help`` here will show the other documentation
formats that may be available (although be aware that some of them require
additional dependencies such as :program:`TeX` or :program:`LaTeX`).


Live Preview
""""""""""""

While writing documentation in a markup format such as `reStructuredText`_, it
is very helpful to preview the formatted version of the text.
While it is possible to simply run the ``make html`` command periodically, a
more seamless workflow of this is available.
Within the :file:`docs/` directory is a small Python script called
:file:`sphinx_server.py` that can simply be called with:

.. prompt:: bash

    poetry run python sphinx_server.py

This will run a small process that watches the :file:`docs/` folder contents,
as well as the source files in :file:`src/nrtk/`, for changes.
:command:`make html` is re-run automatically when changes are detected.
This will serve the resulting HTML files at http://localhost:5500.
Having this URL open in a browser will provide you with an up-to-date
preview of the rendered documentation.

Installing with OpenCV
----------------------
One of the optional packages for nrtk is OpenCV. OpenCV is required for
:py:mod:`~nrtk.impls.perturb_image.generic.cv2.blur` perturbers and
:ref:`Optical Transfer Functions <Optical Transfer Function Examples>`. To give users the option
to use either ``opencv-python`` or ``opencv-python-headless``,
nrtk has the ``graphics`` and ``headless`` extras for ``opencv-python`` and
``opencv-python-headless`` respectively. The following commands will install
the ``opencv-python`` version.

For :command:`pip`:

.. prompt:: bash

    pip install nrtk[graphics]

For :command:`conda-forge`:

.. prompt:: bash

    conda install -c conda-forge nrtk-graphics

For `Poetry`_:

.. prompt:: bash

    poetry install --sync --extras graphics


To install the ``opencv-python-headless`` version, replace ``graphics``
with ``headless`` in the above commands.


.. _Pip-install-upgrade: https://pip.pypa.io/en/stable/reference/pip_install/#cmdoption-U
.. _Poetry: https://python-poetry.org
.. _Poetry-installation: https://python-poetry.org/docs/#installation
.. _Poetry-usage: https://python-poetry.org/docs/basic-usage/
.. _Poetry-poetrylock: https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock
.. _Poetry-dependencies: https://python-poetry.org/docs/pyproject/#dependencies-and-dev-dependencies
.. _Sphinx: http://sphinx-doc.org/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
