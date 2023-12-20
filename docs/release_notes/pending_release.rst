Pending Release Notes
=====================

Updates / New Features
----------------------

Documentation

* Added an Introduction section to provide background and conceptual information about ``nrtk``.

* Added an inheritance diagram which visualizes the layout of ``nrtk`` interfaces and
  implementations and included in interface documentation page.

* Added ``Getting Started`` page.

* Added Read the Docs configuration files

* Added a style sheet to guide future documentation and text updates.


Interfaces

* Added a ``GenerateClassifierBlackboxResponse`` interface for generating response
  curves with given perturber factories, classifier, and scorer.

* Added a ``ScoreClassification`` interface that takes in ground-truth and predicted
  classifications and generates scores based on a given metric.

Security

* Upgraded ``Pillow>=10.0``
