* Added fault-tolerant plugin discovery via ``nrtk.interfaces._plugfigurable``.
  ``PerturbImage.get_impls()`` and ``PerturbImageFactory.get_impls()`` now
  gracefully skip broken third-party entrypoints instead of crashing. This
  works around an upstream scipy 1.17 ``array_api_compat`` bug that causes
  ``TypeError`` during ``smqtk_plugins`` entrypoint scanning on Python 3.11+.
