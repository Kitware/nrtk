#################################
Creating a Public Release Request
#################################

Shared Storage
==============
This `Google Drive folder
<https://drive.google.com/drive/folders/1wZbX-yP5y8AYyudVjMinC2Md6g4UeGv8>`_
is used to store submitted and approved public release requests.

* The `Submitted <https://drive.google.com/drive/folders/1q-JBegvkJIxVngYO6KHM7TitHuzfButN>`_
  folder is used to store public release request packages that are or are
  intended to be submitted, and not approved yet.

* The `Approved <https://drive.google.com/drive/folders/1E5NtKw8lbWLPPQ2VnSFnAdg4AsB_Vcsf>`_
  folder is used to store public release request packages that *have* been
  approved. These folders should have a copy-of/reference-to the approving
  document.

Folders underneath :code:`Submitted` and :code:`Approved` folders generally take the form
of the request submission date in :code:`YYYY-MM-DD` format, e.g.
:code:`2022-12-02`.

Submission Components
=====================
The following are the critical components to a public release submission that
we should place into an applicable sub-folder under :code:`Submitted`:

* `Summary Document`_

* `Code Package`_

Summary Document
----------------
Create a google doc under the appropriate "Submitted" sub-folder and title it
"Summary".
This should contain a summarizing paragraph or two, and then a list of changes
since the last public release.
Basically the change notes since the last public release request submission.
Draw from the release notes in the :code:`docs/release_notes/` files.

The summary document to be sent is usually a Word document (:code:`.docx`)

`Example document. <https://docs.google.com/document/d/1Z3Lh7aXHAKwUNBE9U3kAzXjWyR7hekbaSXvfr4gqBes/edit#>`_

Code Package
------------
This should be a git archive of the codebase at the point which we want to get
released.

In the repository, checkout the branch that is to be submitted for release
approval.
Ideally this is the current :code:`main`/:code:`master` branch.

Create a tag to mark where in the repository the submission, and future
approval, applies to.
This is important as we will need to know where this is to know what is
appropriate to expose when the request is approved.
The tag should be of the form :code:`public-release-request-YYYYMMDD`,
obviously replacing the :code:`YYYYMMDD` with the request submission date.
This date should additionally match the dates mentioned above for the
submission request folder to just keep everything sane and matching.

Create a git archive while on this tag with.
Consider the following example, setting the :code:`YYYYMMDD` with the
appropriate value to match tag to archive:

.. prompt:: bash

   YYYYMMDD="20221202"
   git archive \
       --format tar.gz \
       -o "nrtk-${YYYYMMDD}-$(git rev-parse HEAD).tar.gz" \
       public-release-request-${YYYYMMDD}
