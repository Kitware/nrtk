Steps of the Release Process
============================
Three types of releases are expected to occur:
  - major
  - minor
  - patch

See the :file:`CONTRIBUTING.md` file for information on how to contribute features
and patches.
The following process should apply when any release that changes the version
number occurs.

Create and Merge Version Update Branch
--------------------------------------

This step of the release process depends on whether the release is considered to be major or
minor, or
if it is a patch release.

Major and Minor Releases
^^^^^^^^^^^^^^^^^^^^^^^^
Major and minor releases may add one or more trivial or non-trivial features
and functionalities.

1. Create a new branch off of the ``main`` named something like
   ``update-to-v{NEW_VERSION}``, where ``NEW_VERSION`` is the new ``X.Y``
   version.

   a. Use the ``scripts/update_release_notes.sh`` script to update the project
      version number, create ``docs/release_notes/v{NEW_VERSION}.rst``, and add
      a new pending release notes stub file.

      .. code-block:: bash

         $ # When creating a major release
         $ ./scripts/update_release_notes.sh major
         $ # OR when creating a minor release
         $ ./scripts/update_release_notes.sh minor

   b. Add a descriptive paragraph under the title section of
      ``docs/release_notes/v{NEW_VERSION}.rst`` summarizing this release.

   c. Update ``docs/_static/switcher.json`` (see `Update Version Switcher`_).

   d. If the release is at JATIC Maturity Level I or higher, apply the
      maturity label (see `Maturity Labeling`_).

2. Push the created branch to the upstream repository, not your fork (this is
   an exception to the normal forking workflow).

3. Create a pull/merge request for this branch with ``release`` as the merge
   target. This is to ensure that everything passes CI testing before making
   the release. If there is an issue, then topic branches should be made and
   merged into this branch until the issue is resolved.

4. Get an approving review.

5. Merge the pull/merge request into the ``release`` branch.

6. Tag the resulting merge commit.
   See `Tag new version`_ below for how to do this.

7. As a repository administrator, merge the ``release`` branch into ``main``
   locally and push the updated ``main`` to upstream. (Replace "upstream"
   in the example below with your applicable remote name.)

   .. code-block:: bash

      $ git fetch --all
      $ git checkout upstream/main
      $ git merge --log --no-ff upstream/release
      $ git push upstream main

8. `Draft a new release on GitLab`_ for the new version.

Patch Release
^^^^^^^^^^^^^
A patch release should only contain fixes for bugs or issues with an existing
release.
No new features or functionality should be introduced in a patch release.
As such, patch releases should only ever be based on an existing release point
(git tag).

This list assumes we are creating a new patch release off of the *latest*
release version, i.e. off of the ``release`` branch.
If a patch release for an older release version is being created, see the
`Patching an Older Release`_ section.

1. Create a new branch off of the ``release`` branch named something like
   ``update-to-v{NEW_VERSION}``, where ``NEW_VERSION`` is the target ``X.Y.Z``,
   including the bump in the patch (``Z``) version component.

   a. Use the ``scripts/update_release_notes.sh`` script to update the project
      version number, create ``docs/release_notes/v{NEW_VERSION}.rst``, and add
      a new pending release notes stub file.

      .. code-block:: bash

         $ ./scripts/update_release_notes.sh patch

   b. Add a descriptive paragraph under the title section of
      ``docs/release_notes/v{NEW_VERSION}.rst`` summarizing this release.

   c. Update ``docs/_static/switcher.json`` (see `Update Version Switcher`_).

   d. If the release is at JATIC Maturity Level I or higher, apply the
      maturity label (see `Maturity Labeling`_).

   e. Ensure the ``python-versions`` and the tox environment matrix in
   ``nightly:tox:pytest`` (in ``.gitlab-ci/.gitlab-prod.yml``) are updated for
   the new release.

2. Push the created branch to the upstream repository, not your fork (this is
   an exception to the normal forking workflow).

3. Create a pull/merge request for this branch with ``release`` as the merge
   target. This is to ensure that everything passes CI testing before making
   the release. If there is an issue, then topic branches should be made and
   merged into this branch until the issue is resolved.

4. Get an approving review.

5. Merge the pull/merge request into the ``release`` branch.

6. Tag the resulting merge commit.
   See `Tag new version`_ below for how to do this.

7. As a repository administrator, merge the ``release`` branch into ``main``
   locally and push the updated ``main`` to upstream. (Replace "upstream"
   in the example below with your applicable remote name.)

   .. code-block:: bash

      $ git fetch --all
      $ git checkout upstream/main
      $ git merge --log --no-ff upstream/release
      $ git push upstream main

8. `Draft a new release on GitLab`_ for the new version.

9. `Update Mature Release CI`_ for the new version.

Patching an Older Release
"""""""""""""""""""""""""
When patching a major/minor release that is not the latest version, a branch
needs to be created based on the release version being patched to integrate the
specific patches into.
This branch should be prefixed with ``release-`` to denote that it is a release
integration branch, e.g. ``release-v1.2.19`` (where ``19`` is the incremented
patch version number).
Patch topic-branches should be based on this ``release-...`` branch.
When all fix branches have been integrated, follow the `Patch Release`_ section
above, replacing ``release`` branch references (merge target) to be the
``release-...`` integration branch.
Step 6 should be to merge this release integration branch into ``release``
first, and *then* ``release`` into ``main``, if applicable (some exceptional
patches may only make sense for specific versions and don't warrant integration
into upstream main).

Update Version Switcher
-----------------------
``docs/_static/switcher.json`` powers the version dropdown shown at the top
of the rendered docs site. Entries are tracked at the major/minor level
only; the URL on each entry points at the latest patch in that line. A
patch release must not introduce new functionality, so its docs are
functionally equivalent to the major/minor release it descends from.

When releasing:

* **Major / minor releases**: add a new entry with the new major/minor
  string (e.g. ``"1.1"``) and a URL pointing at the new release tag.

* **Patch releases**: do not add a new entry. Update the URL on the
  existing major/minor entry so it points at the new patch's tag (e.g.
  ``1.0`` URL bumped from ``/en/v1.0.3/`` to ``/en/v1.0.4/``).

Entries follow the pattern:

.. code-block:: json

   {
       "version": "1.0",
       "url": "https://nrtk.readthedocs.io/en/v1.0.4/"
   }

Maturity Labeling
-----------------
Per RS-4 of the program standards, releases at JATIC Maturity Level I or
higher must carry a consistent label, and releases below Maturity I must
not. When a release is at Maturity I or higher, apply the label in three
places:

a. Add an admonition at the top of
   ``docs/release_notes/v{NEW_VERSION}.rst``:

   .. code-block:: rst

      .. admonition:: JATIC Maturity I
         :class: note

         Release v{NEW_VERSION} has been assessed against and meets the
         requirements associated with JATIC Maturity Level I.

b. Title the GitLab release ``v{NEW_VERSION} — JATIC Maturity I`` when
   :ref:`drafting a new release on GitLab <release-process-draft-gitlab>`.

c. Update the maturity admonition in the ``Acknowledgment`` section of the
   docs landing page (``docs/index.rst``) and in the ``Acknowledgment``
   section of ``README.md`` so the version reference matches this release.
   The admonition wording should mirror step (a) — i.e., reference
   ``Release v{NEW_VERSION}`` — so the same text appears across all three
   surfaces.

Replace ``I`` with the appropriate Roman numeral if the release is at a
higher maturity level.

Tag New Version
---------------
Release branches are tagged in order to record where in the git tree a
particular release refers to.
All release tags should be in the history of the ``release`` and ``main``
branches (barring exceptional circumstances).

We prefer to use local ``git tag`` commands to create the release version
tag, pushing the tag to upstream.
The version tag should be applied to the merge commit resulting from the
above described ``update-to-v{NEW_VERSION}`` topic-branch ("the release").

See the example commands below, replacing ``HASH`` with the appropriate git
commit hash, and ``UPSTREAM`` with the appropriate remote name.
We also show how to use `Poetry's version command`_ to consistently access the
current, just-updated package version.

.. code-block:: bash

   $ git checkout HASH
   $ VERSION="v$(poetry version -s)"
   $ git tag -a "$VERSION" -F docs/release_notes/"$VERSION".rst
   $ git push UPSTREAM "$VERSION"

.. _release-process-draft-gitlab:

Draft a New Release on GitLab
-----------------------------
After creating and pushing a new version tag, a GitLab "release" should be
made.

a. Navigate to the GitLab `Releases page`_ for the nrtk repository.

b. Click the "Create a new release" button (or go `here
   <gitlab-new-release-page_>`_).

c. Select from the "Tag name" dropdown the tag version just created and
   pushed

d. Enter the version number as the title, e.g. "v1.2.3".

e. Select the release date.

f. Copy and paste the release notes for this version into the release notes
   field.

g. Click the "Create Release" button to create the GitLab release!

In the future, this may be automated.
See the `appropriate GitLab documentation <gitlab-release-using-cicd_>`_ for
more details.


Update Mature Release CI
------------------------
Once a new release is created, the nightly CI should be updated for the new
release.

a. Create a new schedule under **Build → Pipeline schedules → New schedule**
   set to run daily with ``NRTK_RELEASE_REF`` set to the new GitLab release
   tag and ``SCHEDULE_TARGET`` set to ``prod``.

b. Disable (or delete) the nightly pipeline for previous mature release.


.. _Poetry's version command: https://python-poetry.org/docs/cli/#version
.. _Releases page: https://gitlab.jatic.net/jatic/kitware/nrtk/-/releases
.. _gitlab-new-release-page: https://gitlab.jatic.net/jatic/kitware/nrtk/-/releases/new
.. _gitlab-release-using-cicd: https://docs.gitlab.com/ee/user/project/releases/#creating-a-release-by-using-a-cicd-job
