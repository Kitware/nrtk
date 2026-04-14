* Update to latest DSO CI Components

* Update ruff dependency as part of DSO component updates and resolve related linting failures.

* Added programmatically defined "optional" marker to pytests for DSO Component updates. This marker
  includes all other markers, except for the "notebooks" marker.

* Added ``require_marker`` pytest fixture to properly exclude tests that must be run with a
  marker when pytest is called without specifying a marker.
