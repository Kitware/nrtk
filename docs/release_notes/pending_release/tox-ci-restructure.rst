* Added ``tox.ini`` with isolated test environments for each optional dependency
  combination (``core``, ``opencv``, ``albumentations``, ``pillow``, ``skimage``,
  ``pybsm``, ``waterdroplet``, ``diffusion``, ``maite``, ``tools``, ``doctests``).
  Each environment installs only its required extras and runs the corresponding
  pytest marker-filtered tests.

* Restructured CI test jobs in ``.gitlab-ci/.gitlab-test.yml``: split the
  monolithic ``tox:pytest`` job into separate per-Python-version jobs
  (``tox:pytest:py3.10`` through ``tox:pytest:py3.13``) and added per-environment
  tox jobs for finer-grained optional dependency testing.
