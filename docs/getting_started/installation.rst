.. :auto introduction:

Installation
============

There are multiple ways to obtain the nrtk package.
The simplest is to install via the :command:`pip` command.
Alternatively, you can install via :command:`conda-forge` command.
For local development, you can use `Poetry`_.

nrtk installation has been tested on Unix and Linux systems.

.. :auto introduction:

.. :auto install-commands:

.. _pip:

From :command:`pip`
-------------------

.. prompt:: bash

    pip install nrtk

.. _conda:

From :command:`conda-forge`
---------------------------

.. prompt:: bash

    conda install -c conda-forge nrtk

.. :auto install-commands:

.. :auto from-source:

From Source
-----------
The following assumes `Poetry`_ (`installation`_ and `usage`_) is already installed.

.. note::
  nrtk requires Poetry 2.2 or higher.

`Poetry`_ acts as a comprehensive tool for dependency management, virtual environment handling,
and package building. It streamlines development by automating tasks like dependency resolution,
ensuring consistent environments across different machines, and simplifying the packaging and
publishing of Python projects. Unlike the previous options, `Poetry`_ will not only allow developers
to install any extras they need, but also install multi-dependency groups like nrtk's
`docs <https://github.com/Kitware/nrtk/blob/main/pyproject.toml#L132>`_,
`tests <https://github.com/Kitware/nrtk/blob/main/pyproject.toml#L147>`_, and
`linting <https://github.com/Kitware/nrtk/blob/7014707c0a531fa63fa6d08d7d6aeba9868f09b4/pyproject.toml#L118>`_ tools.


Be sure to note the following warning from Poetry's own documentation:

.. warning::
  Poetry should always be installed in a dedicated virtual environment to isolate it from the rest of your system.
  It should in no case be installed in the environment of the project that is to be managed by Poetry. This ensures
  that Poetry's own dependencies will not be accidentally upgraded or uninstalled. In addition, the isolated virtual
  environment in which poetry is installed should not be activated for running poetry commands.

If unfamiliar with Poetry, take a moment to familiarize yourself using the above links, to ensure the smoothest
introduction possible.

.. note::
  Poetry installation is only recommended for advanced nrtk users. For most users, :ref:`pip<pip>` or
  :ref:`conda<conda>` installation is sufficient.

.. :auto from-source:

.. :auto quick-start:

Quick Start
^^^^^^^^^^^

.. prompt:: bash

    cd /where/things/should/go/
    git clone https://github.com/kitware/nrtk.git ./
    poetry install

.. :auto quick-start:

.. :auto dev-deps:

Installing Developer Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following installs both core and development dependencies as
specified in the :file:`pyproject.toml` file, with versions specified
(including for transitive dependencies) in the :file:`poetry.lock` file:

.. prompt:: bash

    poetry sync --with linting,tests,docs

.. note::
  Developers should also ensure their enviroment has Git LFS installed
  before their first commit. See the `Git LFS documentation <https://git-lfs.com/>`_
  for more details.

.. :auto dev-deps:

.. :auto build-docs:

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

.. :auto build-docs:

.. :auto live-preview:

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

.. :auto live-preview:

Extras
------

NRTK has multiple optional extras to expand functionality. The list below contains the extra name and a brief
description of the extra.

    **graphics**: installs the graphics version of ``opencv-python``. ``graphics`` or ``headless`` is required for
    :mod:`~nrtk.impls.perturb_image.photometric.blur` perturbers, :ref:`AlbumentationsPerturber`,
    :ref:`RandomRotationPerturber`, and :ref:`RandomScalePerturber`.

    **headless**: installs the headless version of ``opencv-python``. ``graphics`` or ``headless`` is required for
    :mod:`~nrtk.impls.perturb_image.photometric.blur` perturbers, :ref:`AlbumentationsPerturber`,
    :ref:`RandomRotationPerturber`, and :ref:`RandomScalePerturber`.

    **pybsm**: installs `pyBSM <https://pybsm.readthedocs.io/en/latest/index.html>`_. Required for
    :ref:`PybsmPerturber`, :ref:`PybsmOTFPerturber`,
    :ref:`TurbulenceApertureOTFPerturber`, :ref:`JitterOTFPerturber`, :ref:`DetectorOTFPerturber`,
    :ref:`CircularApertureOTFPerturber`, and :ref:`DefocusOTFPerturber`.

    **maite**: installs `MAITE <https://github.com/mit-ll-ai-technology/maite>`_ and its associated dependencies.
    Required for everything in :ref:`Interoperability`.

    **tools**: installs `KWCOCO <https://github.com/Kitware/kwcoco>`_ and
    `Pillow <https://pillow.readthedocs.io/en/stable/>`_. Required for :ref:`COCOJATICObjectDetectionDataset`,
    :ref:`nrtk-perturber`, and :func:`~nrtk.interop.maite.datasets.object_detection.dataset_to_coco`.

    **scikit-image**: installs `scikit-image <https://scikit-image.org/>`_. Required for
    :mod:`~nrtk.impls.perturb_image.photometric.noise` perturbers.

    **Pillow**: installs `Pillow <https://pillow.readthedocs.io/en/stable/>`_. Required for
    :mod:`~nrtk.impls.perturb_image.photometric.enhance` perturbers.

    **albumentations**: installs `nrtk-albumentations <https://github.com/Kitware/nrtk-albumentations>`_. Required for
    :mod:`~nrtk.impls.perturb_image.wrapper.albumentations_perturber`,
    :mod:`~nrtk.impls.perturb_image.geometric.random_rotation_perturber`, and
    :mod:`~nrtk.impls.perturb_image.geometric.random_scale_perturber`.

    **waterdroplet**: installs `scipy <https://scipy.org/>`_ and `numba <https://numba.pydata.org/>`_. Required for
    :mod:`~nrtk.impls.perturb_image.environment.water_droplet_perturber` perturber and utility functions.

    **diffusion**: installs `torch <https://pytorch.org/>`_, `diffusers <https://github.com/huggingface/diffusers>`_,
    `accelerate <https://github.com/huggingface/accelerate>`_, and `Pillow <https://pillow.readthedocs.io/en/stable/>`_.
    Required for :ref:`DiffusionPerturber`.

    **notebook-testing**: installs various dependencies required for running any notebook in ``docs/examples``.

Installing with OpenCV
^^^^^^^^^^^^^^^^^^^^^^
One of the optional packages for nrtk is OpenCV, which is required for
:py:mod:`~nrtk.impls.perturb_image.photometric.blur` perturbers and
:doc:`Optical Transfer Function perturbations </examples/otf_visualization>`.

OpenCV receives dedicated installation guidance due to its unique dual-installation options.
Unlike other optional dependencies that have single-path installations,
OpenCV requires users to choose between ``opencv-python`` (full GUI capabilities) and
``opencv-python-headless`` (minimal, no GUI) versions depending on their deployment environment
and requirements.

To give users the option
to use either ``opencv-python`` or ``opencv-python-headless``, nrtk has the ``graphics`` and ``headless``
extras for ``opencv-python`` and ``opencv-python-headless``, respectively.

``opencv-python-headless`` is a
minimal package version of ``opencv-python`` that contains the core
capabilities of OpenCV, without including any of the GUI-related functionalities.

The following commands will install the ``opencv-python`` version.

For :command:`pip`:

.. prompt:: bash

    pip install nrtk[graphics]

For :command:`conda-forge`:

.. prompt:: bash

    conda install -c conda-forge nrtk-graphics

For `Poetry`_:

.. prompt:: bash

    poetry sync --extras graphics


To install the ``opencv-python-headless`` version, replace ``graphics`` with ``headless`` in the above
commands.

.. _perturber-dependencies:

Perturber Dependencies
----------------------
The following table lists the perturbers and the extra/dependencies required to use them.

.. list-table:: Perturber Dependencies
    :widths: 45 25 30
    :header-rows: 1

    * - Perturber
      - Extra(s) Required
      - Key Dependencies Provided by Extra(s)
    * - **Photometric Perturbers**
      -
      -
    * - :ref:`AverageBlurPerturber`
      - ``graphics`` or ``headless``
      - ``OpenCV``
    * - :ref:`BrightnessPerturber`
      - ``Pillow``
      - ``Pillow``
    * - :ref:`ColorPerturber`
      - ``Pillow``
      - ``Pillow``
    * - :ref:`ContrastPerturber`
      - ``Pillow``
      - ``Pillow``
    * - :ref:`GaussianBlurPerturber`
      - ``graphics`` or ``headless``
      - ``OpenCV``
    * - :ref:`GaussianNoisePerturber`
      - ``scikit-image``
      - ``scikit-image``
    * - :ref:`MedianBlurPerturber`
      - ``graphics`` or ``headless``
      - ``OpenCV``
    * - :ref:`PepperNoisePerturber`
      - ``scikit-image``
      - ``scikit-image``
    * - :ref:`SaltAndPepperNoisePerturber`
      - ``scikit-image``
      - ``scikit-image``
    * - :ref:`SaltNoisePerturber`
      - ``scikit-image``
      - ``scikit-image``
    * - :ref:`SharpnessPerturber`
      - ``Pillow``
      - ``Pillow``
    * - :ref:`SpeckleNoisePerturber`
      - ``scikit-image``
      - ``scikit-image``
    * - **Geometric Perturbers**
      -
      -
    * - :ref:`RandomCropPerturber`
      - ---
      - ---
    * - :ref:`RandomRotationPerturber`
      - ``albumentations``, and (``graphics`` or ``headless``)
      - ``nrtk-albumentations``, ``OpenCV``
    * - :ref:`RandomScalePerturber`
      - ``albumentations``, and (``graphics`` or ``headless``)
      - ``nrtk-albumentations``, ``OpenCV``
    * - :ref:`RandomTranslationPerturber`
      - ---
      - ---
    * - **Environment Perturbers**
      -
      -
    * - :ref:`HazePerturber`
      - ---
      - ---
    * - :ref:`WaterDropletPerturber`
      - ``waterdroplet``
      - ``scipy``, ``numba``
    * - **Optical Perturbers**
      -
      -
    * - :ref:`CircularApertureOTFPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :ref:`DefocusOTFPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :ref:`DetectorOTFPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :ref:`JitterOTFPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :ref:`PyBSMOTFPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :ref:`PyBSMPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :ref:`RadialDistortionPerturber`
      - ---
      - ---
    * - :ref:`TurbulenceApertureOTFPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - **Generative Perturbers**
      -
      -
    * - :ref:`DiffusionPerturber`
      - ``diffusion``
      - ``torch``, ``diffusers``, ``accelerate``, ``Pillow``
    * - **Wrapper Perturbers**
      -
      -
    * - :ref:`AlbumentationsPerturber`
      - ``albumentations``, and (``graphics`` or ``headless``)
      - ``nrtk-albumentations``, ``OpenCV``
    * - :ref:`ComposePerturber`
      - ---
      - ---

.. :auto installation-links:

.. _Poetry: https://python-poetry.org
.. _installation: https://python-poetry.org/docs/#installation
.. _usage: https://python-poetry.org/docs/basic-usage/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html

.. :auto installation-links:
