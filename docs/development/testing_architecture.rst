Testing Architecture
********************

NRTK uses `tox <https://tox.wiki/en/stable/>`_ to run its test suite in isolated
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

Run the linting, formatting and type checks:

.. prompt:: bash

    tox -e precommit

.. tip::
   Replace ``310`` with your Python version (``311``, ``312``, ``313``, or ``314``).

The hooks are declared as bare commands with ``language: system``, so they run
in whatever environment invoked ``pre-commit``. ``poetry run pre-commit run
--all-files`` and a plain ``git commit`` are equivalent ways to run the same
checks; they differ only in which environment the hooks inherit. Whichever you
use has to satisfy :file:`pyproject.toml` including its extras, because
``pyright`` reports real failures against a stale or extras-less environment.
The ``precommit`` tox environment is rebuilt from the declared constraints, so
it cannot drift; a Poetry environment needs keeping in sync:

.. prompt:: bash

    poetry sync --with main,linting,tests,docs --all-extras

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

One factor is not an extra: appending ``-gpu`` (e.g. ``py313-diffusion-gpu``)
leaves the extras alone and passes ``CUDA_VISIBLE_DEVICES`` through instead of
blanking it, which is what every other environment does so results don't depend
on the runner's hardware. The variable still has to be set — a ``-gpu`` environment
run without it is CPU-only like any other. It is used by the GPU CI job, which sets
it explicitly, to exercise the device-transfer paths in ``DiffusionPerturber`` and
the MAITE augmentation wrappers; see :ref:`gpu-driver-compatibility` for the driver
those paths require.

Adding a New Implementation
---------------------------

When adding a perturber (or any class) that depends on an optional package:

1. Add the class to a ``Group`` in the owning module's ``guard()`` call, and to
   its ``if TYPE_CHECKING:`` imports (see :doc:`import_guards`).
2. Add a pytest marker to :file:`pyproject.toml` if the dependency group is
   new (see `Pytest Markers`_).
3. Decorate test classes with ``@pytest.mark.<marker>``.
4. Add a ``conftest.py`` with ``pytest_ignore_collect()`` in the test
   directory (see `Directory-Level Collection Skipping`_).
5. Wire up a new tox environment in :file:`tox.ini` if the dependency group
   is new (see `Tox Configuration`_).

No guard test and no canary test is needed. Both are generated from the ``Group``
you declared in step 1 — see `Import Guard Tests`_ and `Canary Tests`_.


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

.. pytestmark: skip
.. code-block:: python

   import pytest
   from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin

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

A **canary test** calls ``pytest.fail()`` — not ``pytest.skip()`` — when a marker is
selected but the extra behind it is not actually installed. This catches CI
configuration errors where a tox environment is supposed to have an extra but
does not, producing an explicit failure rather than silently skipping every test
that depends on it.

There is nothing to write. ``tests/test_guard_canaries.py`` generates one canary
per ``Group`` that declares extras, carrying the marker for those extras, so
declaring the group is what creates its canary:

.. pytestmark: skip
.. code-block:: python

   CANARIES = [
       pytest.param(
           module,
           group,
           marks=[getattr(pytest.mark, marker) for marker in _markers_for(group.extras)],
           id=f"{module}-{'+'.join(_markers_for(group.extras))}",
       )
       for module, group in DECLARED
       if group.extras
   ]

Markers are named after the extra they require, so the mapping is the identity
with one exception: ``graphics`` and ``headless`` ship the same library, so both
map to the ``opencv`` marker. That exception is the only package-specific fact in
the file.

A group declaring no extras gets no canary — its symbols resolve in every
environment, so the check would assert nothing about how that environment was
built.

Import Guard Tests
------------------

Every module that exposes a conditional implementation declares it through
``nrtk._guard.guard()``, which installs the module's :pep:`562` hooks. That is
what makes the isolation above testable: ``import nrtk`` succeeds regardless of
which extras are installed, and a user only sees an error — naming the extra to
install — when they reach for a perturber whose dependency is missing.
:doc:`import_guards` is the reference for how to declare one; this page covers
how they are tested.

Four files cover the guard, and **none of them are edited when adding an
implementation** — a new ``Group`` is picked up automatically:

``tests/test_guard_declarations.py``
    What each module declares, and what follows from it. Parametrized over every
    ``guard()`` call site.

``tests/test_guard_canaries.py``
    That a selected marker's extras are really installed. One case per ``Group``
    that declares extras (see `Canary Tests`_).

``tests/test_guard.py``
    The guard's own mechanics, against throwaway leaves. Edited only when
    ``nrtk._guard`` itself changes.

``tests/test_import_guards_e2e.py``
    The two claims that only hold across a whole process.

Every module that installs a guard is checked by
``tests/test_guard_declarations.py``, which asserts that what the module
declared actually holds:

- Every extra named in a ``Group`` really exists. The check reads
  :file:`src/nrtk/utils/_extras.yml`, which a pre-commit hook regenerates from
  ``[project.optional-dependencies]`` in :file:`pyproject.toml` — so
  :file:`pyproject.toml` is the source of truth, reached indirectly. Add an extra
  there without running the hook and this check still sees the old list, so a
  ``Group`` naming the new extra is reported as unknown until the file is
  regenerated.
- No symbol is claimed by two groups, and no class is published from two
  modules — re-homing sets ``__module__`` globally, so the second binding would
  silently win.
- The ``if TYPE_CHECKING:`` block and the declaration agree in both directions.
- Every module under ``nrtk.impls`` that publishes implementations has an
  ``smqtk_plugins`` entry, without which discovery has nothing to walk however
  correct the guard is.
- Every declared symbol either resolves and is re-homed, or raises an
  ``ImportError`` naming the extras it needs.

That last check deliberately does no dependency mocking. A symbol is either
importable in the current environment or it is not, so the ``core`` environment
exercises every "extra is missing" path and the ``optional`` environment
exercises every "extra is present" path, across every guarded module, without
touching ``sys.modules``.

It also holds three checks that follow from a declaration but only show up
elsewhere: that plugin discovery never returns a private implementation, that a
serialized config records public paths at every level of nesting, and that no
module imports a guard-bound name from the package containing it.

``tests/test_guard.py`` covers the guard's mechanics instead, against throwaway
leaf modules in ``tests/_utils/guard_leaves/`` — message wording for each
requirement shape, eager versus lazy resolution, and the rule that ``__dir__``
never advertises a name ``__getattr__`` would reject. Nothing there reads the
real package, so it does not grow as implementations are added.

The notebook example tests under
:file:`docs/examples/xaitk_saliency_workflow/notebook_tests/` use their own
``ImportGuardTestsMixin``, which guards notebook-local modules that
``test_guard_declarations.py`` does not walk.

``tests/test_import_guards_e2e.py`` is reserved for the claims that cannot hold
in-process, because they only mean anything in an interpreter that has not
already imported half of nrtk. Each is checked by running a script from
``tests/scripts/guard/`` as a subprocess, which prints a single line the
test asserts on. There are only two, one per script:

``packages_import_without_extras``
    The compliance walk. Every package under ``nrtk`` imports with all optional
    dependencies blocked by a meta-path finder, so the check is real even in the
    all-extras environment.

``discovery_across_opt_in``
    Confirms ``get_impls()`` neither comes back empty nor depends on import order. Walking
    the experimental gate from closed to open in one interpreter: entrypoints stay
    inert while it is shut, a package imported first advertises nothing, discovery
    returns the implementation once opted in, it reaches it through the entrypoint
    rather than the earlier import, and the already-imported package advertises it
    afterwards. Those five observations are one script because each depends on the
    state the previous one left behind, and none survive an interpreter where
    ``tests/conftest.py`` has already opted in.

    The subject is a fixture, ``tests/_utils/guard_plugin``, registered under
    ``smqtk_plugins`` by a throwaway distribution the script writes at runtime. Using
    a real implementation would tie a check of the guard's machinery to whatever NRTK
    currently ships: the only one that fits — experimental, and importable with no
    extras — is temporary, and once it graduates there may be none left. That
    a real module is registered at all is checked separately, by
    ``tests/test_guard_declarations.py``.

The two scripts under ``tests/scripts/guard/`` are real ``.py`` files
rather than source passed to ``python -c``, so that ruff and pyright check them,
and so that a failure points at a real line. Reach for one only when import order or a
process-global gate is the thing under test; anything else belongs in one of the
other two files.


Tox Configuration
=================

The :file:`tox.ini` at the repository root defines the full test matrix.

Default Environments
--------------------

The default matrix is:

.. code-block:: ini

   py{310,311,312,313,314}-{core,opencv,albumentations,pillow,waterdroplet,skimage,diffusion,pybsm,hcipy,maite,tools,optional,doctests}

Each environment:

1. Creates an isolated virtualenv (``skip_install = true``, dependencies
   installed explicitly via ``deps``).
2. Installs ``nrtk`` in editable mode with only the extras for that factor.
3. Installs the ``tests`` dependency group (pytest, pytest-cov, pytest-xdist,
   syrupy, etc.).
4. Sets ``CUDA_VISIBLE_DEVICES`` to empty (prevents GPU contention).
5. Writes coverage to a per-environment file (``.coverage.<envname>``).
6. Runs ``pytest -m "<marker>" -n auto`` with JUnit XML output.

In CI, ``PYTEST_ADDOPTS`` sets ``--maxprocesses`` from the runner's tags:
small CPU runners allow up to 1 worker, medium CPU runners up to 2, and
large CPU runners up to 8. Notebook jobs additionally cap the count at 4,
while retaining any lower runner limit. Other runner tags use a cap of 4.
Tox passes ``PYTEST_ADDOPTS`` through to pytest; local runs retain automatic
CPU detection without these CI limits unless the variable is set explicitly.

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
   `papermill <https://papermill.readthedocs.io/en/stable/>`_. This environment builds
   a local wheel of ``nrtk`` and installs it (simulating a PyPI install) so
   notebooks exercise the same code path users would see.

``precommit``
   Runs every hook in :file:`.pre-commit-config.yaml` over all files:
   `ruff <https://docs.astral.sh/ruff/>`_ lint and format,
   `sphinx-lint <https://github.com/sphinx-contrib/sphinx-lint>`_,
   `pyright <https://github.com/microsoft/pyright>`_ (both internal type
   checking and ``--verifytypes`` public API completeness), and the hook that
   regenerates :file:`src/nrtk/utils/_extras.yml` from :file:`pyproject.toml`.
   This replaced the separate ``ruff``, ``pyright`` and ``sphinx``
   environments, so there is one definition of "the checks" rather than one per
   CI job. The environment installs the package with every extra, which the
   ``pyright`` hooks need in order to resolve optional dependencies. The name
   is spelled without a hyphen because tox splits environment names on hyphens
   into factors: ``pre-commit`` would be read as the two factors ``pre`` and
   ``commit`` rather than as a literal name.


How CI Uses Tox
===============

The GitLab CI pipeline uses the same :file:`tox.ini`, ensuring that local and
CI execution are identical. A shared ``.tox-setup`` base
(in :file:`.gitlab-ci/.gitlab-shared.yml`) installs tox and is extended by
both the test and quality stages.

Test Stage
----------

Defined in :file:`.gitlab-ci/.gitlab-test.yml`:

1. **Parallel test matrix** — For each Python version (3.10–3.14), CI runs
   all tox factors in parallel as separate jobs. Each job invokes
   ``tox -e py<version>-<factor>`` and uploads ``.coverage.*``
   artifacts and JUnit XML reports. Jobs are assigned to different runner
   tags based on resource requirements:

   - ``small-cpu``: core, opencv, albumentations, pillow, skimage
   - ``medium-cpu``: pybsm, hcipy, maite, tools, waterdroplet
   - ``autoscaler``: diffusion, optional, doctests, notebooks
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

Defined in :file:`.gitlab-ci/.gitlab-quality.yml`, this is a single
``precommit`` job running ``tox -e precommit``. Every check lives in
:file:`.pre-commit-config.yaml` — ruff lint and format, sphinx-lint, pyright
internal type checking, and pyright ``--verifytypes`` for public API
completeness — so the same command reproduces CI exactly.


Numba Parallelization Note
--------------------------

The ``pybsm`` and ``waterdroplet`` environments disable ``pytest-xdist``
parallelization by passing ``-n0`` (overriding the default ``-n auto``).
Numba parallelizes with a thread pool inside a single process, so under
``pytest-xdist`` every worker process brings up a pool of its own and the
machine ends up oversubscribed by a factor of the worker count.

That pool is sized from the detected CPU count, which on a workstation means a
test run takes the whole machine even with ``-n0``. ``[testenv]`` lists
``NUMBA_NUM_THREADS`` in ``pass_env``, so capping it is just:

.. prompt:: bash

    NUMBA_NUM_THREADS=4 tox -e py310-waterdroplet

Every environment inherits that ``pass_env``, including the standalone ones.
The cap is worth reaching for even below the core count: capping
``waterdroplet`` at 4 threads took it from 198s to 11s on a 20-core machine.
These tests call many short-lived kernels rather than a few long ones, so the
per-call cost of synchronizing the pool dominates, and that cost grows with the
number of threads in it. ``optional`` and ``doctests`` run the same numba-backed
tests and benefit the same way.


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

    tox -r -e papermill -- docs/examples/end_to_end_overview.ipynb docs/examples/end_to_end_overview.ipynb

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
- **Target what you need.** The full matrix spans 5 Python versions × 12
  factors (60 environments). During development, use ``tox -e py310-core``
  or ``tox -f py310`` rather than running the entire matrix.
- **Coverage files.** Each environment writes its coverage data to a
  separate file (``.coverage.<envname>``). Combine them with
  ``tox -e coverage`` before generating a report.
- **Tox is not a Poetry dependency.** Install it separately with
  ``pipx install tox``.
