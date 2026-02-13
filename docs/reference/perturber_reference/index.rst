########################
User Input Requirements
########################

This section summarizes the required and optional inputs for each NRTK perturbation, organized by functional category.
For dependency and installation requirements, see the :ref:`perturber-dependencies` table.

NRTK perturbers are organized into six functional categories based on their purpose:

- :doc:`Photometric Perturbers <photometric>`:
  Modify visual appearance (color, brightness, blur, noise)
- :doc:`Geometric Perturbers <geometric>`:
  Alter spatial positioning (rotation, scaling, cropping, translation)
- :doc:`Environment Perturbers <environment>`:
  Simulate atmospheric effects (haze, water droplets)
- :doc:`Optical Perturbers <optical>`:
  Model physics-based sensor and optical phenomena
- :doc:`Generative Perturbers <generative>`:
  Apply AI-based transformations (e.g. diffusion models)
- :doc:`Utility Perturbers <utility>`:
  Enable composition and third-party library integration

.. toctree::
   :maxdepth: 1
   :hidden:

   photometric
   geometric
   environment
   optical
   generative
   utility
