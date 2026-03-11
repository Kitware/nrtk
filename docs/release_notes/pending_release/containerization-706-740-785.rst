* Redesigned container CI pipeline with separate build jobs for branch
  (``build-branch``), main (``build-main``), and release (``build-release``)
  builds. Branch and main builds install nrtk from a locally built wheel;
  release builds install from PyPI.

* Restructured ``Dockerfile`` into multi-stage build with two targets:
  ``build-from-source`` (wheel-based) and ``build-from-pypi``. Removed Poetry
  dependency in favor of ``pip install`` with ``python -m build``.

* Switched to CPU-only PyTorch in container images, reducing image size from
  ~9.2 GB to ~2.7 GB.

* Added container image tagging scheme: release tags match the nrtk version
  (``X.Y.Z``, ``X.Y``, ``X``, ``latest``). Development tags use ``main`` or
  ``<branch-slug>``.

* Added SBOM generation (Syft), vulnerability scanning (Trivy), cosign image
  signing, and SBOM attestation to all build jobs. Scan results are published
  as GitLab container scanning artifacts.

* Added ``verify`` job to pull, verify cosign signature, and smoke-test the
  built image.

* Replaced ``cleanup`` job with a nightly scheduled sweep that queries all
  Harbor artifacts, compares tags against active GitLab branches, and removes
  stale branch images and their cosign artifacts.

* Added stale artifact cleanup in build jobs to remove old cosign signatures
  and SBOM attestations when overwriting an existing tag.

* Switched ``.dockerignore`` to a whitelist approach, only including files
  needed to build the wheel (``src/``, ``pyproject.toml``, ``README.md``,
  ``LICENSE``).

* Added container documentation (``docs/containers/nrtk-perturber.rst``) with
  image tag reference, usage instructions, input arguments, error codes, and
  cosign verification commands.

* Added ``.trivyignore`` for vulnerability scan exceptions.
