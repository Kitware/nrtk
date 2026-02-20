.. :auto introduction:

Advanced Installation
=====================

This page covers advanced installation options for NRTK, including conda,
optional perturber dependencies, and development setup from source
using `Poetry`_.

For basic installation, ``pip install nrtk`` is all you need
(see :doc:`Getting Started </getting_started/quickstart>`).

.. note::
   nrtk has been tested on Unix-based systems, including Linux, macOS, and WSL.

.. :auto introduction:

.. :auto install-commands:

.. _pip:
.. _conda:

pip & conda
-----------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: **Prefer pip?**

      NRTK can be installed via pip from `PyPI <https://pypi.org/project/nrtk/>`_.

      .. prompt:: bash

          pip install nrtk[extra1,extra2,...]

   .. grid-item-card:: **Working with conda?**

      NRTK can be installed via conda from `conda-forge <https://github.com/conda-forge/nrtk-feedstock>`_.

      .. prompt:: bash

          conda install -c conda-forge nrtk

See :ref:`perturber-dependencies` below for the full list of optional dev extras
and which perturbers they enable.

.. :auto install-commands:

.. _perturber-dependencies:

Perturber Dependencies
----------------------
The following table lists each perturber and the ``pip install nrtk[extra]``
extras required to use them. Install any combination of extras as needed for
your use case (e.g., ``pip install nrtk[pybsm,headless]``).

.. note::
   Perturbers that require OpenCV list ``graphics`` or ``headless`` as their extra.
   Use ``graphics`` for full GUI capabilities (``opencv-python``) or ``headless``
   for minimal, no-GUI environments (``opencv-python-headless``).
   The conda package includes all optional dependencies by default.

.. list-table:: Perturber Dependencies
    :widths: 45 25 30
    :header-rows: 1

    * - Perturber
      - Extra(s) Required
      - Key Dependencies Provided by Extra(s)
    * - **Photometric Perturbers**
      -
      -
    * - :class:`~nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber`
      - ``graphics`` or ``headless``
      - ``OpenCV``
    * - :class:`~nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber`
      - ``pillow``
      - ``Pillow``
    * - :class:`~nrtk.impls.perturb_image.photometric.enhance.ColorPerturber`
      - ``pillow``
      - ``Pillow``
    * - :class:`~nrtk.impls.perturb_image.photometric.enhance.ContrastPerturber`
      - ``pillow``
      - ``Pillow``
    * - :class:`~nrtk.impls.perturb_image.photometric.blur.GaussianBlurPerturber`
      - ``graphics`` or ``headless``
      - ``OpenCV``
    * - :class:`~nrtk.impls.perturb_image.photometric.noise.GaussianNoisePerturber`
      - ``skimage``
      - ``scikit-image``
    * - :class:`~nrtk.impls.perturb_image.photometric.blur.MedianBlurPerturber`
      - ``graphics`` or ``headless``
      - ``OpenCV``
    * - :class:`~nrtk.impls.perturb_image.photometric.noise.PepperNoisePerturber`
      - ``skimage``
      - ``scikit-image``
    * - :class:`~nrtk.impls.perturb_image.photometric.noise.SaltAndPepperNoisePerturber`
      - ``skimage``
      - ``scikit-image``
    * - :class:`~nrtk.impls.perturb_image.photometric.noise.SaltNoisePerturber`
      - ``skimage``
      - ``scikit-image``
    * - :class:`~nrtk.impls.perturb_image.photometric.enhance.SharpnessPerturber`
      - ``pillow``
      - ``Pillow``
    * - :class:`~nrtk.impls.perturb_image.photometric.noise.SpeckleNoisePerturber`
      - ``skimage``
      - ``scikit-image``
    * - **Geometric Perturbers**
      -
      -
    * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomCropPerturber`
      - ---
      - ---
    * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomRotationPerturber`
      - ``albumentations``, and (``graphics`` or ``headless``)
      - ``nrtk-albumentations``, ``OpenCV``
    * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomScalePerturber`
      - ``albumentations``, and (``graphics`` or ``headless``)
      - ``nrtk-albumentations``, ``OpenCV``
    * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomTranslationPerturber`
      - ---
      - ---
    * - **Environment Perturbers**
      -
      -
    * - :class:`~nrtk.impls.perturb_image.environment.HazePerturber`
      - ---
      - ---
    * - :class:`~nrtk.impls.perturb_image.environment.WaterDropletPerturber`
      - ``waterdroplet``
      - ``scipy``, ``numba``
    * - **Optical Perturbers**
      -
      -
    * - :class:`~nrtk.impls.perturb_image.optical.otf.CircularAperturePerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :class:`~nrtk.impls.perturb_image.optical.otf.DefocusPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :class:`~nrtk.impls.perturb_image.optical.otf.DetectorPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :class:`~nrtk.impls.perturb_image.optical.otf.JitterPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :class:`~nrtk.impls.perturb_image.optical.PybsmPerturber`
      - ``pybsm``
      - ``pyBSM``
    * - :class:`~nrtk.impls.perturb_image.optical.radial_distortion_perturber.RadialDistortionPerturber`
      - ---
      - ---
    * - :class:`~nrtk.impls.perturb_image.optical.otf.TurbulenceAperturePerturber`
      - ``pybsm``
      - ``pyBSM``
    * - **Generative Perturbers**
      -
      -
    * - :class:`~nrtk.impls.perturb_image.generative.DiffusionPerturber`
      - ``diffusion``
      - ``torch``, ``diffusers``, ``accelerate``, ``Pillow``
    * - **Utility Perturbers**
      -
      -
    * - :class:`~nrtk.impls.perturb_image.AlbumentationsPerturber`
      - ``albumentations``, and (``graphics`` or ``headless``)
      - ``nrtk-albumentations``, ``OpenCV``
    * - :class:`~nrtk.impls.perturb_image.ComposePerturber`
      - ---
      - ---

.. :auto from-source:

From Source
-----------
The following assumes `Poetry`_ is already installed. Otherwise, please refer to Poetry
`installation`_ and `usage`_ before proceeding.

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

Clone & Install
^^^^^^^^^^^^^^^

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
--------------------------
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
^^^^^^^^^^^^

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

.. :auto installation-links:

.. _Poetry: https://python-poetry.org
.. _installation: https://python-poetry.org/docs/#installation
.. _usage: https://python-poetry.org/docs/basic-usage/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html

.. :auto installation-links:
