Testing Architecture
********************

NRTK uses `tox <https://tox.wiki>`_ to run its test suite in isolated
environments. This page explains the architecture, the reasoning behind it,
and how the pieces fit together — from import guards in the source code, to
pytest markers on test classes, to the tox environments that tie it all
together, to the GitLab CI pipeline that runs them per commit.


.. _running-tests:

Quick Start
===========

.. note::
   Tox is not included in the Poetry dependency groups. Install it separately
   before running the commands below:

   .. prompt:: bash

       pipx install tox

Common Commands
---------------

Run the core tests (no optional extras):

.. prompt:: bash

    tox -e py310-core

Run tests for a specific extra:

.. prompt:: bash

    tox -e py310-pybsm

Run all test groups for one Python version:

.. prompt:: bash

    tox -f py310

List all available environments:

.. prompt:: bash

    tox list

.. tip::
   Replace ``310`` with your Python version (``311``, ``312``, or ``313``).

The first run will be slow as tox creates a fresh virtualenv for every
environment. Subsequent runs reuse cached environments and are significantly
faster.

.. _env-extras-table:

Environment-to-Extras Mapping
-----------------------------

Each tox environment name follows the pattern ``py<version>-<factor>``,
where the factor directly corresponds to one of the project's optional extras
(e.g., ``py310-pybsm`` installs ``nrtk[pybsm]``). The special ``core``
factor installs nrtk with no extras, and ``doctests`` installs all of them.
Run ``tox list`` to see every available environment.

Adding a New Implementation
---------------------------

When adding a perturber (or any class) that depends on an optional package:

1. Add an import guard to the source module (see `Import Guards`_ below).
2. Add a pytest marker to :file:`pyproject.toml` if the dependency group is
   new (see `Pytest Markers`_).
3. Decorate test classes with ``@pytest.mark.<marker>``.
4. Add a ``conftest.py`` with ``pytest_ignore_collect()`` in the test
   directory (see `Directory-Level Collection Skipping`_).
5. Add a canary test for the new dependency group (see `Canary Tests`_).
6. Add an ``ImportGuardTestsMixin`` subclass (see `Import Guard Mixin Tests`_).
7. Wire up a new tox environment in :file:`tox.ini` if the dependency group
   is new (see `Tox Configuration`_).


Why Isolated Test Environments?
===============================

NRTK's value proposition includes a **modular dependency model**: users can
``pip install nrtk[pybsm]`` to get only the pyBSM optical perturbers, or
deploy ``nrtk`` core into a restricted environment with zero optional
dependencies. This means the codebase must work correctly with *any subset*
of its 10 optional extras installed.

To guarantee this, each test group runs in an environment that has *only* the
extras that group needs. If a test accidentally imports ``cv2`` in an
environment that only has ``pillow`` installed, it fails — catching a real
dependency leak that would affect users.


Import Guards
=============

Every module that depends on an optional package wraps its imports in a
``try``/``except ImportError`` block at module level:

.. code-block:: python

   # src/nrtk/impls/perturb_image/photometric/blur.py (simplified)

   _CV2_CLASSES = ["AverageBlurPerturber", "GaussianBlurPerturber", "MedianBlurPerturber"]
   __all__: list[str] = []
   _import_error: ImportError | None = None

   try:
       from nrtk.impls.perturb_image.photometric._blur.average_blur_perturber import (
           AverageBlurPerturber as AverageBlurPerturber,
       )
       # ... other imports ...
       __all__ += _CV2_CLASSES
   except ImportError as _ex:
       _import_error = _ex

   def __getattr__(name: str) -> None:
       if name in _CV2_CLASSES:
           msg = (
               f"{name} requires the `graphics` or `headless` extra. "
               f"Install with: `pip install nrtk[graphics]` or `pip install nrtk[headless]`"
           )
           if _import_error is not None:
               msg += f"\n\n... upstream error:\n  {type(_import_error).__name__}: {_import_error}"
           raise ImportError(msg)
       raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

This pattern achieves two things:

1. **Safe imports** — ``import nrtk`` never fails regardless of which extras
   are installed. Users only see an error when they try to *use* a perturber
   whose dependency is missing.

2. **Actionable errors** — The ``ImportError`` message tells the user exactly
   which extra to install. If the extra *is* installed but a transitive
   dependency failed, the original upstream error is surfaced as well.


Pytest Markers
==============

Every test class (or function) is decorated with a **pytest marker** that
declares which optional dependency it requires. These markers are defined in
:file:`pyproject.toml`:

.. code-block:: toml

   # pyproject.toml (excerpt — see the file for the full list)
   [tool.pytest.ini_options]
   markers = [
       "core: Run tests that only require core functionality",
       "opencv: Run tests that require the graphics or headless extra",
       "pybsm: Run tests that require the pybsm extra",
       # ... one marker per optional extra ...
   ]

Tests apply these as class-level decorators:

.. code-block:: python

   @pytest.mark.opencv
   class TestGaussianBlurPerturber(PerturberTestsMixin):
       ...

Each tox environment runs ``pytest -m "<marker>"`` so that only the tests
matching the installed extras are collected. Some marker expressions use
boolean logic to handle overlapping dependencies:

- ``opencv``: ``pytest -m "opencv and not albumentations"`` — runs
  OpenCV-only tests, excluding those that also require Albumentations.
- ``albumentations``: ``pytest -m "albumentations and opencv"`` — runs tests
  that need both Albumentations and OpenCV together.
- ``maite``: ``pytest -m "maite and not tools"`` — runs MAITE interop tests,
  excluding CLI/entrypoint tests that also need the ``tools`` extra.
- ``tools``: ``pytest -m "maite and tools"`` — runs only the entrypoint tests
  that require both extras.


Safety Nets
===========

Beyond markers, the test suite includes two additional safety mechanisms:

Directory-Level Collection Skipping
------------------------------------

Each test directory containing optional-dependency tests has a
``conftest.py`` that implements ``pytest_ignore_collect()``:

.. code-block:: python

   # tests/impls/perturb_image/photometric/blur/conftest.py

   def pytest_ignore_collect() -> bool | None:
       """Skip this directory if blur perturbers are not importable."""
       try:
           from nrtk.impls.perturb_image.photometric.blur import (
               AverageBlurPerturber, GaussianBlurPerturber, MedianBlurPerturber,
           )
           del AverageBlurPerturber, GaussianBlurPerturber, MedianBlurPerturber
       except ImportError:
           return True
       return None

This prevents collection errors if a marker is misapplied — the directory is
silently skipped rather than causing a test run failure.

Canary Tests
------------

Each optional-dependency group includes a **canary test** that attempts to
import the expected classes and calls ``pytest.fail()`` (not ``pytest.skip()``)
if the import fails:

.. code-block:: python

   @pytest.mark.opencv
   def test_opencv_public_imports() -> None:
       """Canary test: FAIL if opencv marker is used but blur perturbers can't be imported."""
       try:
           from nrtk.impls.perturb_image.photometric.blur import (
               AverageBlurPerturber, GaussianBlurPerturber, MedianBlurPerturber,
           )
           del AverageBlurPerturber, GaussianBlurPerturber, MedianBlurPerturber
       except ImportError as e:
           pytest.fail(
               f"Running with opencv marker but blur perturbers not importable: {e}. "
               f"Ensure graphics or headless extra is installed.",
           )

This catches CI configuration errors where a tox environment is supposed to
have an extra installed but doesn't, producing an explicit failure rather than
silently skipping all the tests that depend on it.

Import Guard Mixin Tests
------------------------

Import guard behavior itself is tested via the ``ImportGuardTestsMixin``
(in ``tests/_utils/import_guard_tests_mixin.py``). These tests are marked
``core`` — they run in the core environment with no extras installed. The
mixin temporarily injects ``None`` into ``sys.modules`` to simulate a missing
dependency, then verifies:

- Guarded classes raise the expected ``ImportError`` with correct
  installation instructions.
- Guarded classes are excluded from the module's ``__all__``.
- Always-available classes remain importable.
- Unknown attribute access raises ``AttributeError``.


Tox Configuration
=================

The :file:`tox.ini` at the repository root defines the full test matrix.

Default Environments
--------------------

The default matrix is:

.. code-block:: ini

   py{310,311,312,313}-{core,opencv,albumentations,pillow,waterdroplet,skimage,diffusion,pybsm,maite,tools,doctests}

Each environment:

1. Creates an isolated virtualenv (``skip_install = true``, dependencies
   installed explicitly via ``deps``).
2. Installs ``nrtk`` in editable mode with only the extras for that factor.
3. Installs the ``tests`` dependency group (pytest, pytest-cov, pytest-xdist,
   syrupy, etc.).
4. Sets ``CUDA_VISIBLE_DEVICES`` to empty (prevents GPU contention).
5. Writes coverage to a per-environment file (``.coverage.<envname>``).
6. Runs ``pytest -m "<marker>" -n auto`` with JUnit XML output.

See the :ref:`Environment-to-Extras Mapping <env-extras-table>` section for
more on how factors map to extras.

Standalone Environments
-----------------------

These are **not** included in the default ``tox`` run or factor-filtered
runs (``tox -f py310``). They must be invoked explicitly because they serve
specialized purposes.

``py{310,...}-notebooks``
   Runs ``pyright`` type checking over the example notebooks. This
   environment has its own heavily-pinned dependency set (torch, ultralytics,
   numpy, etc.) that is intentionally kept in :file:`tox.ini` rather than
   :file:`pyproject.toml` to avoid dependency resolution conflicts with the
   rest of the project.

``coverage``
   Combines per-environment ``.coverage.*`` artifact files into a single
   report, generates a Cobertura XML file, and enforces a 90% line-coverage
   threshold (a JATIC SDP requirement). Run ``tox -f py310`` first, then
   ``tox -e coverage``.

``papermill``
   Executes Jupyter notebooks end-to-end using
   `papermill <https://papermill.readthedocs.io/>`_. This environment builds
   a local wheel of ``nrtk`` and installs it (simulating a PyPI install) so
   notebooks exercise the same code path users would see.

``ruff``
   Runs the `ruff <https://docs.astral.sh/ruff/>`_ linter and formatter in
   check mode. Combines what were previously two separate CI jobs
   (``ruff-lint`` and ``ruff-format``) into a single invocation.

``pyright``
   Runs `pyright <https://github.com/microsoft/pyright>`_ type checking.
   By default (``tox -e pyright``), it runs internal type checking across the
   source tree. Pass ``--verifytypes`` via posargs for public API completeness
   checking:

   .. prompt:: bash

       tox -e pyright -- --verifytypes nrtk --ignoreexternal src/nrtk

``sphinx``
   Lints Sphinx/RST documentation using
   `sphinx-lint <https://github.com/sphinx-contrib/sphinx-lint>`_.


How CI Uses Tox
===============

The GitLab CI pipeline uses the same :file:`tox.ini`, ensuring that local and
CI execution are identical. A shared ``.tox-setup`` base
(in :file:`.gitlab-ci/.gitlab-shared.yml`) installs tox and is extended by
both the test and quality stages.

Test Stage
----------

Defined in :file:`.gitlab-ci/.gitlab-test.yml`:

1. **Parallel test matrix** — For each Python version (3.10–3.13), CI runs
   all tox factors in parallel as separate jobs. Each job invokes
   ``tox -e py<version>-<factor>`` and uploads ``.coverage.*``
   artifacts and JUnit XML reports. Jobs are assigned to different runner
   tags based on resource requirements:

   - ``small-cpu``: core, opencv, albumentations, pillow, skimage
   - ``medium-cpu``: pybsm, maite, tools, waterdroplet
   - ``autoscaler``: diffusion, doctests, notebooks
   - ``single-gpu``: generative notebook (requires GPU)

2. **Per-version coverage combine** — After all factor jobs for a Python
   version finish, a follow-up job combines their ``.coverage.*`` files
   into a single ``.coverage.<version>`` file.

3. **Final coverage report** — A final job runs ``tox -e coverage`` to
   combine all per-version coverage files, generate a Cobertura XML report,
   and enforce the 90% threshold. This combined report is used by GitLab's
   coverage visualization.

4. **Notebook execution** — Notebooks are run via ``tox -e papermill`` in
   separate jobs, triggered manually on merge requests and automatically on
   scheduled pipelines.

Quality Stage
-------------

Defined in :file:`.gitlab-ci/.gitlab-quality.yml`:

1. **Ruff** — Runs ``tox -e ruff`` (linting + format checking).
2. **Pyright internal** — Runs ``tox -e pyright`` for full source type checking.
3. **Pyright external** — Runs ``tox -e pyright -- --verifytypes ...`` and
   validates 100% public API type completeness. The completeness validation
   logic (score parsing, threshold enforcement, artifact generation) remains
   in the CI script.
4. **Sphinx lint** — Runs ``tox -e sphinx`` to lint RST documentation.


Numba Parallelization Note
--------------------------

The ``pybsm`` and ``waterdroplet`` environments disable ``pytest-xdist``
parallelization by passing ``-n0`` (overriding the default ``-n auto``).
Numba's JIT compiler uses its own process-level parallelization internally,
which conflicts with ``pytest-xdist`` spawning multiple worker processes.


Updating Notebooks Before Committing
=====================================

Jupyter notebooks checked into the repository should have up-to-date cell
outputs and clean metadata. The ``papermill`` tox environment handles both
in a single command — it re-executes every cell and then runs ``nbstripout``
to strip personal metadata (kernel name, execution timestamps, widget state)
while keeping the cell outputs and execution counts intact.

To update a notebook in place, pass the **same path** as both the input and
output arguments:

.. prompt:: bash

    tox -r -e papermill -- docs/examples/nrtk_tutorial.ipynb docs/examples/nrtk_tutorial.ipynb

.. tip::
   The ``-r`` flag forces tox to recreate the environment. This is used in CI
   to ensure a clean state, but you can omit it locally for faster iteration
   if you haven't changed the ``nrtk`` source since the last run.

Under the hood, this environment:

1. **Builds a local wheel** of ``nrtk`` from the working tree
   (``python -m build --wheel``).
2. **Installs the wheel** with ``--no-index``, so any ``%pip install nrtk``
   commands inside the notebook install the local build instead of pulling
   from PyPI. This ensures notebook outputs reflect your current code.
3. **Executes the notebook** with ``papermill``, which runs every cell top to
   bottom and fails on any exception.
4. **Strips metadata** with ``nbstripout --keep-output --keep-count``,
   removing personal/environment metadata while preserving the rendered
   outputs and execution counts that readers expect to see.

After the command completes, the notebook file is updated in your working
tree. Review the diff and commit it alongside any related code changes.

.. note::
   CI runs notebooks with ``/dev/null`` as the output path — it only
   validates that execution succeeds, it does not update the committed
   notebook. Keeping notebooks up to date is the developer's responsibility
   before pushing.


Practical Tips
==============

- **First run is slow.** Tox creates a fresh virtualenv for every
  environment. Subsequent runs reuse cached environments and are
  significantly faster.
- **Target what you need.** The full matrix spans 4 Python versions × 11+
  factors (44+ environments). During development, use ``tox -e py310-core``
  or ``tox -f py310`` rather than running the entire matrix.
- **Coverage files.** Each environment writes its coverage data to a
  separate file (``.coverage.<envname>``). Combine them with
  ``tox -e coverage`` before generating a report.
- **Tox is not a Poetry dependency.** Install it separately with
  ``pipx install tox``.
