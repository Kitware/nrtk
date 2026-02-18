* Fixed notebook rendering being out of sync with releases by redesigning
  the ``tox`` ``papermill`` environment to build a local wheel and install it
  with ``--no-index``, so notebook ``%pip install`` commands use the local
  build instead of PyPI. Added ``nbstripout`` to strip personal metadata from
  executed notebooks. Changed ``base_python`` to ``py313``.

* Fixed ``print_extras_status()`` showing ``unknown`` for package versions by
  updating ``_try_import()`` to use ``importlib.metadata.version()`` instead
  of the module ``__version__`` attribute. Changed the readthedocs URL from a
  version-pinned path to ``/en/stable/``.

* Standardized notebook setup cells across all 17 CI-tracked notebooks:
  consistent ``%pip install`` format, correct per-notebook extras, removed
  unnecessary ``numpy<2.0`` pins, and consolidated scattered install cells
  into a single setup cell per notebook.

* Fixed diffusion notebook hardcoding ``device="cuda"`` which caused failures
  on CPU-only runners. The notebook now uses ``DiffusionPerturber``'s built-in
  auto-detection (CUDA when available, CPU otherwise).

* Changed ``jatic-perturbations-saliency`` notebook using ``JitterPerturber``
  to ``AverageBlurPerturber``.

* Fixed PDF docs CI job failing by switching ``latex_engine`` to
  ``lualatex`` which handles Unicode/emoji characters natively,
  installing ``texlive-luatex`` and ``fonts-freefont-ttf`` in CI,
  using ``-pdflua`` latexmk flag, and overriding Sphinx's default
  ``fontpkg`` to use Latin Modern fonts bundled with texlive.
