* Fixed nightly scheduled CI pipeline wiping the coverage badge by removing
  the schedule exclusion from test and coverage jobs.

* Updated CI to run the full MR pipeline (tests, quality, docs, security,
  containers) on nightly scheduled builds. Mirror and publish stages remain
  unaffected.

* Fixed compliance trigger job running on tag-initiated publish pipelines by
  scoping its rules to MR events and default branch pushes only.
