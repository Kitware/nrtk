Getting Started
===============

The Natural Robustness Toolkit (NRTK) helps you test AI model robustness by
simulating real-world operational conditions. Follow these steps to install nrtk,
run your first perturbation, and connect your scenario to the right tools.

----

Step 1: Install nrtk
---------------------

.. tab-set::

   .. tab-item:: pip

      nrtk can be installed via pip from `PyPI <https://pypi.org/project/nrtk/>`_.

      .. warning::
          The recommended way to install nrtk via ``pip`` is to use a virtual environment. To learn
          more, see `creating virtual environments in the Python Packaging User Guide
          <https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments>`_.

      .. code-block:: bash

          $ pip install nrtk


   .. tab-item:: conda

      nrtk can be installed via conda from `conda-forge <https://github.com/conda-forge/nrtk-feedstock>`_.

      .. warning::
          The recommended way to install nrtk via ``conda`` is to use a virtual environment. To learn
          more, see `creating environments in the conda documentation
          <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_.

      .. code-block:: bash

          $ conda install -c conda-forge nrtk

.. seealso::
  For details on optional extras, installing from source, or developer setup, see
  :doc:`Installation </getting_started/installation>`.

Step 2: Run a Sample Perturbation
----------------------------------

:class:`~nrtk.impls.perturb_image.environment.HazePerturber` requires no additional dependencies beyond
the base nrtk install, making it ideal for a first test.

.. pytestmark: core
.. code-block:: python

   from nrtk.impls.perturb_image.environment import HazePerturber
   import numpy as np
   from PIL import Image

   # Load your image as a numpy array
   INPUT_IMG_FILE = 'docs/images/input.jpg'
   image = np.array(Image.open(INPUT_IMG_FILE))

   # Create a haze perturber with medium haze strength
   perturber = HazePerturber(factor=1.0)

   # Apply the perturbation
   img_out, _ = perturber(image=image)

Step 3: See the Results
-----------------------

The ``factor`` parameter controls haze intensity. Here's what different levels look like:

.. list-table::
   :widths: 33 33 33
   :header-rows: 1

   * - Light (factor=0.5)
     - Medium (factor=1.0)
     - Heavy (factor=1.5)
   * - .. image:: /images/operational_risk_modules/haze_light.png
          :width: 200px
     - .. image:: /images/operational_risk_modules/haze_medium.png
          :width: 200px
     - .. image:: /images/operational_risk_modules/haze_heavy.png
          :width: 200px

.. seealso::
  These images demonstrate how NRTK simulates real-world visibility degradation.
  For parameter details, see the :class:`~nrtk.impls.perturb_image.environment.HazePerturber` API.
  For operational context, see the
  :doc:`Haze Simulation Module </explanations/operational_risk_modules/haze>`.

Step 4: Map Your Risk to the Right Tool
-----------------------------------------

Have a specific operational condition or test scenario in mind?
Use the Interactive Operational Risk Matrix to discover which perturbations
represent your mission environment and conditions.

.. card::

   The Risk Matrix maps real-world operational risks (weather, sensor limitations,
   platform vibration) to the NRTK perturbations that simulate them.

   +++

   .. button-ref:: /explanations/risk_factors
      :color: primary
      :outline:

      Interactive Risk Matrix →

.. toctree::
   :hidden:

   installation
   first_perturbation
   where_to_go_next
