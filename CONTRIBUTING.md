# Contributing to `NRTK`

## Making a Contribution
Here we describe at a high level how to contribute to `NRTK`.
See the [`NRTK` README](README.md) file for additional information.

1.  Navigate to the official NRTK repository maintained [on GitHub](
    https://github.com/Kitware/nrtk).

2.  Fork NRTK into your GitHub user namespace and clone that onto
    your system.

3.  Create a topic branch, edit files and create commits:

        $ git checkout -b <branch-name>
        $ <edit things>
        $ git add <file1> <file2> ...
        $ git commit

    * Included in your commits should be an addition to the
      `docs/release_notes/pending_release.rst` file (or the relevant patch
      release notes file).
      This addition should be a short, descriptive summary of the update,
      feature or fix that was added.
      This is generally required for merger approval.

4.  Push topic branch with commits to your fork in GitHub:

        $ git push origin HEAD -u

5.  Visit the Kitware NRTK GitHub, browse to the "Pull requests" tab
    and click on the "New pull request" button in the upper-right.
    Click on the "Compare across forks" link, browse to your fork and browse to
    the topic branch for the pull request.
    Finally, click the "Create pull request" button to create the request.

`NRTK` uses GitHub for code review and GitHub Actions for continuous
testing.
New pull requests trigger Continuous Integration workflows (CI) when the merge
target is the `main` or `release`-variant branch.
All checks/tests must pass before a PR can be merged by an individual with the
appropriate permissions.

GitHub/GitLab: The GitHub repository is a mirror of a private GitLab repository,
so once the PR can be merged, it will not be merged via GitHub. A maintainer will
push the branch to the GitLab repository and do the final merging there, which
will then get mirrored out to main.

We use Sphinx for manual and automatic API [documentation](docs).

### Jupyter Notebooks
When adding or modifying a Jupyter Notebook in this repository, consider the
following:
* Notebooks should be included in the appropriate CI workflow and be runnable
  in that environment in a timely manner.
* Notebooks should also be runnable when executed in CI.
  This often requires a cell that performs ``pip install ...`` commands to bring
  in the appropriate dependencies (including `nrtk` itself) into the
  at-first empty environment.

#### Notebook CI
This repository has set up a CI workflow to execute notebooks to ensure their
continued functionality, avoid bit-rot and avoid violation of established
use-cases.
When contributing a Jupyter notebook, as an example or otherwise, a reference
should be added to this CI workflow ([located here ~L165](
.gitlab-ci.yml)) to enable its inclusion in the CI
testing. Additionally, it should be added to the static typing notebook
workflow (located in `test/examples/test_notebooks.py`).

To that end, in developing the notebook, consider its execution in this CI
environment:
* should be concise in its execution time to not stall or time-out the CI
  workflow.
* should be light on computational resources to not exceed what is provided in
  the CI environment.

### Contribution Release Note Exceptions
When a new contribution is fixing a bug or minor issue with something that has
been recently contributed, it may be the case that no additional release notes
are needed since they would add redundancy to the document.

For example, let's say that a recent contribution added a feature `Foo` and
an appropriate release note for that feature.
If a bug with that feature is quickly noticed and fixed in a follow-on
contribution that does not impact how the feature is summarized in the
release notes, then the release-notes check on that follow-on contribution may
be ignored by the reviewers of the contribution.

Generally, a reviewer will assume that a release note is required unless the
contributor makes a case that the check should be ignored.
This will be considered by reviewers on a case-by-case basis.

## Class Naming Philosophy
For classes that define a behavior, or perform a transformation of a
subject, we choose to follow the "Verb-Noun" style of class naming.
The motivation to prioritize the verb first is because of our heavy use of
interface classes to define API standards and how their primary source of
implementation variability is in "how" the verb is achieved.
The noun subject of the verb usually describes the input provided, or output
returned, at runtime.
Another intent of this take on naming positively impacts intuitive
understanding of class meaning and function for user-facing interfaces and
utilities.

Class names should represent the above as accurately, yet concisely as
possible and appropriately differentiate from other classes defined here.
If there are multiple classes that perform similar behaviors but on different
subjects, it is important to distinguish the subject of the verb in the
naming.

## Non-public Contributions
It is reasonable to expect that some extensions to this package may not be
desired to be released into the public scope.
This package makes use of a plugin framework to allow for derivative packages
to define their own interface implementations, such that they are discoverable
when such a package is present in the same python environment as this package.
SMQTK-Core documentation found [here][smqtk_plugin_reference] describes how
such a derivative package would expose their implementations such that they
would be discoverable by the plugin framework.

If such originally non-public contributions ever do become publicly
releasable, they may continue to live in their original homes, or may be
contributed into this codebase via a pull request (see [above](
#Making-a-Contribution) and [our review process](docs/review_process.rst)).


[smqtk_plugin_reference]: https://smqtk-core.readthedocs.io/en/stable/plugins_configuration.html#creating-an-interface-and-exposing-implementations
