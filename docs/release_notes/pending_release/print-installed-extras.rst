* Added an internal utility script to update the ``_extras.yml`` file from
  ``pyproject.toml``.

* Added a pre-commit hook to run the utility script to ensure that the
  ``_extras.yml`` file is always up-to-date with the extras defined in
  ``pyproject.toml``.

* Added an internal utility script to print the installed extras and their
  dependencies.
