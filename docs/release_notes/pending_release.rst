Pending Release Notes
=====================

Updates / New Features
----------------------

* Removed perturber interface restriction which required that input image dimensions be maintained.
  Note perturbers which modify image dimensions (including rotations) should be used with caution as
  scoring can be impacted if groundtruth isn't similarly transformed.

Fixes
-----
