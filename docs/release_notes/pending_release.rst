Pending Release Notes
=====================

Updates / New Features
----------------------

Build

* New minimum supported python changed to ``python = "^3.9"`` due to 3.8 EOL.

Dependencies

* Updated python minimum requirement to 3.9 (up from 3.8.1) due to 3.8 EOL. This included updates to certain
  dependencies with bifurcations, an update to pinned versions for development/CI, and removal of 3.8 from CI.

Documentation

* Updated README to include a reference to the ``nrtk-jatic`` package.

* Restored and improved review process documentation.

* Added ``sphinx-click`` as a dev docs dependency.

* Fixed sphinx linting errors.

Implementations

* Updated default sx and sy values to 0 for JitterOTFPerturber

Fixes
-----
