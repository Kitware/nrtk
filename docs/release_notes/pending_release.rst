Pending Release Notes
=====================

Updates / New Features
----------------------

Examples

* Added an example notebook exploring the current state of several augmentation
  tools as well as the usability of the JATIC Toolbox augmentation protocol.

Interfaces

* Added a ``PerturbImage`` interface for taking an image stimulus and
  generating a perturbed image.

* Added a ``ScoreDetection`` interface that takes in ground-truth and predicted
  BBox-label pairs and generates scores based on a given metric.

Implementations

* Added several ``PerturbImage`` implementations

  * ``SaltPerturber``, ``PepperPerturber``, ``SaltAndPepperPerturber``: Adds
    salt and/or pepper noise to given image stimulus.

  * ``GaussianPerturber``: Adds Gaussian-distributed additive noise to given
    image stimulus.

  * ``SpecklePerturber``: Adds multiplicative (Gaussian) noise to given image
    stimulus.

  * ``NOPPerturber``: Serves as a pass-through NOP perturber to test interface
    functionality.

* Added a ``ScoreDetection`` implementation, ``NOPScorer`` which serves
  as a pass-through NOP scorer to test interface functionality.

Fixes
-----
