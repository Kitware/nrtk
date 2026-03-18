* Migrate CI quality stage jobs (ruff, pyright, sphinx-lint) from ``poetry run``
  to tox for local reproducibility and consistency with the test stage.

* Add ``.tox-setup`` CI base to centralize the tox version pin, replacing
  duplicated installs in ``.test-setup`` and ``.quality-setup``.

* Combine ``ruff-lint`` and ``ruff-format`` CI jobs into a single ``ruff`` job.

* Migrate final ``tox:coverage:`` CI job from ``poetry run coverage`` to
  ``tox -e coverage``.
