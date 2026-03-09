Installation
============

This page covers installation options beyond the basic PyPI/conda install shown
in the :doc:`Getting Started </getting_started/quickstart>` section, including
installing from source. NRTK has been tested on Unix-based systems, including
Linux, macOS, and WSL.

.. attention::
   **Not all perturbers are available with the base PyPI install** — many require
   optional third-party libraries such as pyBSM, OpenCV, or Pillow. With PyPI, you
   can selectively install only the extras that your workflow requires (see the
   :ref:`perturber-requirements` table below). The conda-forge package includes all
   optional dependencies by default.

.. _pip:
.. _conda:

Installing nrtk
---------------

.. seealso::
  See :ref:`perturber-requirements` below for the full list of optional extras
  and which perturbers they enable.

.. tab-set::

  .. tab-item:: pip

    nrtk can be installed via pip from `PyPI <https://pypi.org/project/nrtk/>`_.

    .. warning::
        The recommended way to install nrtk via ``pip`` is to use a virtual environment. To learn
        more, see `creating virtual environments in the Python Packaging User Guide
        <https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments>`_.

    .. code-block:: bash

        $ pip install nrtk[extra1,extra2,...]

  .. tab-item:: conda

    nrtk can be installed via conda from `conda-forge <https://github.com/conda-forge/nrtk-feedstock>`_.

    .. warning::
        The recommended way to install nrtk via ``conda`` is to use a virtual environment. To learn
        more, see `creating environments in the conda documentation
        <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_.

    .. code-block:: bash

        $ conda install -c conda-forge nrtk

  .. tab-item:: From source

    Install from source with `Poetry`_ when you are contributing to nrtk, need to
    build the documentation locally, or want to run the test suite.

    .. warning::
      Poetry installation is only recommended for advanced nrtk users. For most users, pip or conda installation is
      sufficient.

    The following assumes `Poetry`_ is already installed. Otherwise, please refer to Poetry
    `installation`_ and `usage`_ before proceeding.

    .. note::
      nrtk requires Poetry 2.2 or higher.

    .. rubric:: About Poetry

    `Poetry`_ acts as a comprehensive tool for dependency management, virtual environment handling,
    and package building. It streamlines development by automating tasks like dependency resolution,
    ensuring consistent environments across different machines, and simplifying the packaging and
    publishing of Python projects. `Poetry`_ not only allows developers to install any extras they need,
    but also install multi-dependency groups like nrtk's docs, tests, and linting tools.

    **Be sure to note the following warning from Poetry's own documentation**:

    .. warning::
      Poetry should always be installed in a dedicated virtual environment to isolate it from the rest of your system.
      It should in no case be installed in the environment of the project that is to be managed by Poetry. This ensures
      that Poetry's own dependencies will not be accidentally upgraded or uninstalled. In addition, the isolated virtual
      environment in which poetry is installed should not be activated for running poetry commands.

    If unfamiliar with Poetry, take a moment to familiarize yourself using the above links, to ensure the smoothest
    introduction possible.

    .. rubric:: Clone and Install

    .. code-block:: bash

        $ cd /where/things/should/go/
        $ git clone https://github.com/kitware/nrtk.git ./
        $ poetry install

    .. rubric:: Installing Developer Dependencies

    The following installs both core and development dependencies as
    specified in the :file:`pyproject.toml` file, with versions specified
    (including for transitive dependencies) in the :file:`poetry.lock` file:

    .. code-block:: bash

        $ poetry sync --with linting,tests,docs

    .. note::
      Developers should also ensure their environment has Git LFS installed
      before their first commit. See the `Git LFS documentation <https://git-lfs.com/>`_
      for more details.

    .. rubric:: Installing Extras

    To install an extra group(s) to enable a perturbation, add the ``--extras`` flag to
    your install command, e.g.:

    .. code-block:: bash

        $ poetry sync --with linting,tests,docs --extras "extra1 extra2 ..."

    .. rubric:: Building the Documentation

    The documentation for NRTK is maintained as a collection of
    `reStructuredText`_ documents in the :file:`docs/` folder of the project.
    The :program:`Sphinx` documentation tool can process this documentation
    into a variety of formats, the most common of which is HTML.

    Within the :file:`docs/` directory is a Unix :file:`Makefile` (for Windows
    systems, a :file:`make.bat` file with similar capabilities exists).
    This :file:`Makefile` takes care of the work required to run :program:`Sphinx`
    to convert the raw documentation to an attractive output format.
    For example, calling the command below will generate
    HTML format documentation rooted at :file:`docs/_build/html/index.html`.

    .. code-block:: bash

        $ poetry run make html


    Calling the command ``make help`` here will show the other documentation
    formats that may be available (although be aware that some of them require
    additional dependencies such as :program:`TeX` or :program:`LaTeX`).

    .. seealso::
      Developers looking to contribute to NRTK should check out our
      :doc:`additional development resources </development/index>`.

.. _perturber-requirements:

Perturber Requirements
----------------------
The following table lists each perturber and the extras required to use them. Install any
combination of extras as needed for your use case (e.g., ``pip install nrtk[pybsm,headless]``).

.. note::
   Perturbers that require OpenCV list ``graphics`` or ``headless`` as their extra.
   Use ``graphics`` for full GUI capabilities (``opencv-python``) or ``headless``
   for minimal, no-GUI environments (``opencv-python-headless``).
   The conda package includes all optional dependencies by default.

Photometric Perturbers
^^^^^^^^^^^^^^^^^^^^^^
.. dropdown:: Modify visual appearance (color, brightness, blur, noise)

  .. list-table::
      :widths: 45 20 25 30
      :header-rows: 1

      * - Perturber
        - Required Inputs
        - Extra(s) Required
        - Key Dependencies Provided by Extra(s)
      * - :class:`~nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber`
        - Image (RGB, Grayscale)
        - ``graphics`` or ``headless``
        - ``OpenCV``
      * - :class:`~nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber`
        - Image (RGB, Grayscale)
        - ``pillow``
        - ``Pillow``
      * - :class:`~nrtk.impls.perturb_image.photometric.enhance.ColorPerturber`
        - Image (RGB)
        - ``pillow``
        - ``Pillow``
      * - :class:`~nrtk.impls.perturb_image.photometric.enhance.ContrastPerturber`
        - Image (RGB, Grayscale)
        - ``pillow``
        - ``Pillow``
      * - :class:`~nrtk.impls.perturb_image.photometric.blur.GaussianBlurPerturber`
        - Image (RGB, Grayscale)
        - ``graphics`` or ``headless``
        - ``OpenCV``
      * - :class:`~nrtk.impls.perturb_image.photometric.noise.GaussianNoisePerturber`
        - Image (RGB, Grayscale)
        - ``skimage``
        - ``scikit-image``
      * - :class:`~nrtk.impls.perturb_image.photometric.blur.MedianBlurPerturber`
        - Image (RGB, Grayscale)
        - ``graphics`` or ``headless``
        - ``OpenCV``
      * - :class:`~nrtk.impls.perturb_image.photometric.noise.PepperNoisePerturber`
        - Image (RGB, Grayscale)
        - ``skimage``
        - ``scikit-image``
      * - :class:`~nrtk.impls.perturb_image.photometric.noise.SaltAndPepperNoisePerturber`
        - Image (RGB, Grayscale)
        - ``skimage``
        - ``scikit-image``
      * - :class:`~nrtk.impls.perturb_image.photometric.noise.SaltNoisePerturber`
        - Image (RGB, Grayscale)
        - ``skimage``
        - ``scikit-image``
      * - :class:`~nrtk.impls.perturb_image.photometric.enhance.SharpnessPerturber`
        - Image (RGB, Grayscale)
        - ``pillow``
        - ``Pillow``
      * - :class:`~nrtk.impls.perturb_image.photometric.noise.SpeckleNoisePerturber`
        - Image (RGB, Grayscale)
        - ``skimage``
        - ``scikit-image``

Geometric Perturbers
^^^^^^^^^^^^^^^^^^^^
.. dropdown:: Alter spatial positioning (rotation, scaling, cropping, translation)

  .. list-table::
      :widths: 45 20 25 30
      :header-rows: 1

      * - Perturber
        - Required Inputs
        - Extra(s) Required
        - Key Dependencies Provided by Extra(s)
      * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomCropPerturber`
        - Image (RGB, Grayscale)
        - ---
        - ---
      * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomRotationPerturber`
        - Image (RGB, Grayscale)
        - ``albumentations``, and (``graphics`` or ``headless``)
        - ``nrtk-albumentations``, ``OpenCV``
      * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomScalePerturber`
        - Image (RGB, Grayscale)
        - ``albumentations``, and (``graphics`` or ``headless``)
        - ``nrtk-albumentations``, ``OpenCV``
      * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomTranslationPerturber`
        - Image (RGB, Grayscale)
        - ---
        - ---

Environment Perturbers
^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Simulate atmospheric effects (haze, water droplets)

  .. list-table::
      :widths: 45 20 25 30
      :header-rows: 1

      * - Perturber
        - Required Inputs
        - Extra(s) Required
        - Key Dependencies Provided by Extra(s)
      * - :class:`~nrtk.impls.perturb_image.environment.HazePerturber`
        - Image (RGB, Grayscale)
        - ---
        - ---
      * - :class:`~nrtk.impls.perturb_image.environment.WaterDropletPerturber`
        - Image (RGB, Grayscale)
        - ``waterdroplet``
        - ``scipy``, ``numba``

Optical Perturbers
^^^^^^^^^^^^^^^^^^

.. dropdown:: Model physics-based sensor and optical phenomena

  .. list-table::
      :widths: 45 20 25 30
      :header-rows: 1

      * - Perturber
        - Required Inputs
        - Extra(s) Required
        - Key Dependencies Provided by Extra(s)
      * - :class:`~nrtk.impls.perturb_image.optical.otf.CircularAperturePerturber`
        - Image (RGB, Grayscale)
        - ``pybsm``
        - ``pyBSM``
      * - :class:`~nrtk.impls.perturb_image.optical.otf.DefocusPerturber`
        - Image (RGB, Grayscale)
        - ``pybsm``
        - ``pyBSM``
      * - :class:`~nrtk.impls.perturb_image.optical.otf.DetectorPerturber`
        - Image (RGB, Grayscale)
        - ``pybsm``
        - ``pyBSM``
      * - :class:`~nrtk.impls.perturb_image.optical.otf.JitterPerturber`
        - Image (RGB, Grayscale)
        - ``pybsm``
        - ``pyBSM``
      * - :class:`~nrtk.impls.perturb_image.optical.PybsmPerturber`
        - Image (RGB, Grayscale)
        - ``pybsm``
        - ``pyBSM``
      * - :class:`~nrtk.impls.perturb_image.optical.RadialDistortionPerturber`
        - Image (RGB, Grayscale)
        - ---
        - ---
      * - :class:`~nrtk.impls.perturb_image.optical.otf.TurbulenceAperturePerturber`
        - Image (RGB, Grayscale)
        - ``pybsm``
        - ``pyBSM``

Generative Perturbers
^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Generate perturbations using diffusion models

  .. list-table::
      :widths: 45 20 25 30
      :header-rows: 1

      * - Perturber
        - Required Inputs
        - Extra(s) Required
        - Key Dependencies Provided by Extra(s)
      * - :class:`~nrtk.impls.perturb_image.generative.DiffusionPerturber`
        - Image (converts to RGB)
        - ``diffusion``
        - ``torch``, ``diffusers``, ``accelerate``, ``Pillow``

Utility Perturbers
^^^^^^^^^^^^^^^^^^

.. dropdown:: Enable composition and third-party library integration

  .. list-table::
      :widths: 45 20 25 30
      :header-rows: 1

      * - Perturber
        - Required Inputs
        - Extra(s) Required
        - Key Dependencies Provided by Extra(s)
      * - :class:`~nrtk.impls.perturb_image.AlbumentationsPerturber`
        - Image (format varies by transform)
        - ``albumentations``, and (``graphics`` or ``headless``)
        - ``nrtk-albumentations``, ``OpenCV``
      * - :class:`~nrtk.impls.perturb_image.ComposePerturber`
        - Image (format varies by perturbers)
        - ---
        - ---

.. _Poetry: https://python-poetry.org
.. _installation: https://python-poetry.org/docs/#installation
.. _usage: https://python-poetry.org/docs/basic-usage/
.. _reStructuredText: https://docutils.sourceforge.io/rst.html
