# Contributing to `nrtk`

## Overview
This guide provides instructions for contributing to the `nrtk` project. For additional details, please refer to the [`nrtk` README](README.md).

## Steps to Contribute

1. **Access the Repository**  
   Navigate to the official `nrtk` repository on [JATIC GitLab](https://gitlab.jatic.net/jatic/kitware/nrtk).

2. **Fork and Clone**  
   Fork the repository into your GitLab user namespace and clone it to your local system.

   ```bash
   $ git clone <your-fork-url>
   $ cd nrtk
   ```

3. **Create a Topic Branch**  
   Work on a new topic branch for your contribution:

   ```bash
   $ git checkout -b <branch-name>
   $ <edit files>
   $ git add <file1> <file2> ...
   $ git commit -m "Descriptive commit message"
   ```

   - Ensure that your commits include an update to the `docs/release_notes/pending_release.rst` file (or a relevant patch release notes file).  
   - Provide a concise summary of the feature, update, or fix being added. This is generally required for approval.

4. **Push Your Branch**  
   Push your changes to your forked repository:

   ```bash
   $ git push origin HEAD -u
   ```

5. **Create a Merge Request (MR)**  
   On GitLab, navigate to the "Merge Requests" tab of your forked project and click "New merge request."  

   - Set the source branch to your forked branch and the target branch to the appropriate branch of the `nrtk` repository.  
   - Click "Compare branches and continue" and then "Create merge request."

---

## Continuous Integration (CI) and Code Review
`nrtk` uses GitLab for code reviews and GitLab CI/CD for automated testing.  
- New MRs trigger CI workflows for `master` and `release` branches.  
- All CI checks must pass before an MR can be merged.

We use [Sphinx](https://www.sphinx-doc.org/) for API documentation, both manual and automated.

---

## Guidelines for Jupyter Notebooks
When adding or modifying Jupyter Notebooks:

1. **CI Inclusion**  
   - Ensure notebooks are part of the CI workflow and executable within its environment.  
   - Add references to `.gitlab-ci/.gitlab-test.yml` and to `tests/examples/test_notebooks.py`.

2. **Execution Requirements**  
   - Keep execution concise to avoid CI timeouts.  
   - Include a cell for dependency installation (e.g., `pip install ...`) to ensure proper setup in the CI environment.

3. **Resource Constraints**  
   - Optimize notebooks to use minimal computational resources, aligning with CI environment limitations.

---

## Contribution Release Notes
Typically, every contribution requires an associated release note. Add this to the `docs/release_notes/pending_release.rst` file, summarizing the changes made.

### Exceptions
If your contribution fixes a minor issue in a recent update and does not alter the summary in existing release notes, the requirement may be waived.  
- Contributors should explain why a release note is unnecessary for the reviewers to consider on a case-by-case basis.

---

## Class Naming Philosophy
We adopt a "Verb-Noun" convention for class names. This structure highlights the behavior or transformation a class performs.  
- **Verb**: Indicates the primary action or behavior.  
- **Noun**: Specifies the subject of the action, typically related to input or output.  

### Examples
- Classes performing similar actions on different subjects should clearly differentiate the subjects in their names.  
- Names should be concise and intuitive, aiding user understanding.

## Issue Reporting
If an issue is found, it can be reported on the [Issue Reporting Page](https://github.com/Kitware/nrtk/issues/new). In the description, provide an outline of the problem in as much detail as possible being sure to include any relevant logs and error tracebacks. Once the issue is created, it will be triaged by a maintainer in a timely manner.

## Feature Requests
If a user wants to request a new feature, it can be reported in the same location as the issues. In the issue created for feature requests, begin the title with `FEATURE REQUEST`. The issue should also include a `Use Case` section describing the intended use case of the new feature.