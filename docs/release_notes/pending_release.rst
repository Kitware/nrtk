Pending Release Notes
=====================

Updates / New Features
----------------------

Documentation

* Updated README to include a reference to the ``nrtk-jatic`` package.

Implementations

* Updated default sx and sy values to 0 for JitterOTFPerturber

* Added optional usage of load_database_atmosphere_no_interp in addition
  to existing load_database_atmosphere functionality

Fixes
-----

* Optional dependencies were setup in a way that ``opencv`` was missing when
  ``pybsm`` was installed, this has been fixed so ``opencv`` can be installed
  via ``pybsm`` install or as standalone for ``nrtk`` on its own.
