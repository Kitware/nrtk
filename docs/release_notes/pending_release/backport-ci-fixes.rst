* Migrated CI jobs and container signing to Kubernetes runners, capped pytest
  workers by runner size, and fixed pre-commit checkout trust.

* Aligned notebook execution with Python 3.13 and raised the minimum tox version
  to 4.22 for dependency groups.

* Updated the container Python base image and pip pin to include security fixes.
