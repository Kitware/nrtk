Review Process
**************

The process for reviewing and integrating branches into nrtk is described
below.

For guidelines on contributing, see ``CONTRIBUTING.md``.

For guidelines on the release process, see `Release Process and
Notes`_.

.. _`Release Process and Notes`: release_process.html

.. contents:: The review process consists of the following parts:
   :local:

Merge Request (MR)
==================
An MR is initiated by a user intending to integrate a branch from their forked
repository.
Before the branch is integrated into the nrtk master branch, it must
first go through a series of checks and a review to ensure that the branch is
consistent with the rest of the repository and doesn't contain any issues.

Workflow Status
---------------
The submitter must set the status of their MR:

Draft
^^^^^
Indicates that the submitter does not think that the MR is in a reviewable or
mergeable state.
Once they complete their work and think that the MR is ready to be considered
for merger, they may set the status to ``Open``.

Open
^^^^
Indicates that an MR is ready for review and that the submitter of the MR thinks
that the branch is ready to be merged.
If a review is received that requests substantial changes to the contributed
content, effectively returning the task at hand into a "development" phase, the
MR status should be changed to ``Draft`` (by the author of the MR) to indicate
that such changes are underway.

If the submitter is still working on the MR and simply wants feedback, they
must request it and leave their branch marked as a ``Draft``.

Closed
^^^^^^
Indicates that the MR is resolved or discarded.

Continuous Integration
======================
The following checks are included in the automated portion of the review
process, and are triggered whenever a merge-request is made or changes, a tag is
created, or when the ``main`` branch is updated.
These are run as part of the CI/CD pipeline driven by GitLab CI pipelines, and
defined by the :file:`.gitlab-ci.yml` file.
The success or failure of each may be seen in-line in a submitted MR in the
"Checks" section of the MR view.

Code Style Consistency (``test-py-lint``)
-----------------------------------------
Runs ``flake8`` to quality check the code style.
You can run this check manually in your local repository with
``poetry run flake8``.

Passage of this check is strictly required.

Static Type Analysis (``test-py-typecheck``)
--------------------------------------------
Performs static type analysis.
You can run this check manually in your local repository with ``poetry run
mypy``.

Passage of this check is strictly required.

Documentation Build (``test-docs-build``)
-----------------------------------------
Performs a build of our Sphinx documentation.

Passage of this check is strictly required.

Unit Tests (``test-pytest``)
----------------------------
Runs the unittests created under ``tests/`` as well as any doctests found in
docstrings in the package code proper.
You can run this check manually  in your local repository with ``poetry run
pytest``.

Passage of these checks is strictly required.

Regression Tests
^^^^^^^^^^^^^^^^
Regression test snapshots are generated using
`syrupy <https://github.com/syrupy-project/syrupy>`_. To generate new snapshots,
run ``poetry run pytest --snapshot-update path/to/test_file.py``. Ensure the full
filepath is included so that irrelevant snapshots are not erroneously updated.
Once a snapshot is generated, regression test results will be included in the
``test-pytest`` job.

Code Coverage (``test-coverage-percent``)
-----------------------------------------
This job checks that the lines of code covered by our Unit Tests checks meet or
exceed certain thresholds.

Passage of this check is not strictly required but highly encouraged.

Release Notes Check (``test-release-notes-check``)
--------------------------------------------------
Checks that the current branch's release notes has modifications relative to
the marge target's.

Passage of this check is not strictly required but highly encouraged.

Example Notebooks Execution (``test-notebooks``)
------------------------------------------------
This check executes included example notebooks to ensure their proper
functionality with the package with respect to a merge request.
Not all notebooks may be run, as some may be set up to use too many resources
or run for an extended period of time.

Passage of these checks is strictly required.

Human Review
============
Once the automatic checks are either resolved or addressed, the submitted MR
will need to go through a human review.
Reviewers should add comments to provide feedback and raise potential issues on
logical and semantic details of the contributed content that would otherwise
not be caught by the discrete automatic checks above.
Should the MR pass their review, the reviewer should then indicate that it has
their approval using the GitLab review interface to flag the MR as ``Approved``.

A review can still be requested before the checks are resolved, but the MR must
be marked as a ``Draft``.
Once the MR is in a mergeable state, it will need to undergo a final review to
ensure that there are no outstanding issues.

If an MR is not a draft and has an appropriate amount of approving reviews, it
may be merged at any time.

If an MR fully resolves an issue, the text
``Closes nrtk#<issue number>`` should be included in the MR description,
so that the issue will be automatically closed upon merge.

For internal development, the following review procedure shall be used:

1. The author will label the merge request as "status::for review", move the
   associated issue(s) to the for review column on the issue board, and request
   a review from a peer reviewer. Once the review process begins, the issue(s)
   should remain in the for review column, review status will only be updated
   on the MR itself, unless the scope of the issue significantly changes.

2. Peer reviewer will provide comments or suggested changes and re-label the
   merge request as "status::in progress". If no work is needed, the reviewer
   will instead approve the MR and request a review from a maintainer.

3. Author will address the comments provided by the peer reviewer and then
   re-label the merge request as "status::for review" and re-request a peer
   review.

4. Peer reviewer will repeat the process starting at step 2, as needed.

5. Maintainer will provide comments or suggested changes and re-label the merge
   request as in progress. If the MR needs no work, then the maintainer will
   instead approve the MR and :ref:`resolve the branch<Resolving a Branch>`.

6. If additional changes are requested, the author will address the comments
   provided by the maintainer and then label the MR as ready for review and
   re-request maintainer review; repeating the procedure starting at step 5, as
   needed.

Notebooks
---------
The default preference is that all Jupyter Notebooks be included in execution
of the Notebook CI job (listed under the ``parallel:matrix`` section).
If a notebook is added in the MR, it should be verified that it has been added
to the list of notebooks to be run.
If it has not been, the addition should be requested or for a rationale as to
why it has not been.
Rationale for excluding specific notebooks from the CI job should be added to
the relevant section in ``examples/README.md``.

Resolving a Branch
==================

Merge
-----
Once an MR receives an approving review and is no longer marked as a ``Draft``,
the repository maintainers can merge it, closing the merge request.
It is recommended that the submitter delete their branch after the MR is
merged.

Close
-----
If it is decided that the MR will not be integrated into ``nrtk``, then
it can be closed through GitLab.
