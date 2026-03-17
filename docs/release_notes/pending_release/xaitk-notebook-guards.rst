* Fixed notebook path resolution when running via papermill from the repository
  root.

* Fixed keyword argument and visualization errors in the object detection
  notebook.

* Refactored import guards in the notebook utility modules to align with the
  current codebase patterns.

* Enabled GPU-accelerated PyTorch in notebook install cells.

* Added import guard unit tests for the notebook utility modules under
  ``docs/examples/nrtk_xaitk_workflow/notebook_tests/``.

* Added ``xaitk-notebook-tests`` CI job to run import guard and canary tests
  across Python 3.10–3.13.

* Broadened the ruff per-file-ignores pattern from ``tests/*.py`` to
  ``**/*tests/*.py`` so notebook tests also receive test-file lint exemptions.
