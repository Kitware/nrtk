* Fixed ``mock_missing_deps`` not restoring parent package attributes after
  re-importing modules, causing ``mock.patch`` on dotted paths through guarded
  packages to fail with ``AttributeError`` in subsequent tests.

* Fixed ``TestOtfImportGuard`` not evicting cached ``_pybsm`` submodules,
  allowing the import guard in ``otf.py`` to be bypassed when ``pybsm`` was
  mocked as missing.

* Added ``optional`` tox environment and CI job that runs all non-notebook tests
  in a single session with all extras installed, catching cross-test pollution
  that per-extra environments miss.
