Pending Release Notes
=====================

Updates / New Features
----------------------

Examples

* Added an example notebook exploring the current state of several augmentation
  tools as well as the usability of the JATIC Toolbox augmentation protocol.

* Added an example notebook demonstrating NRTK perturber functionality.

Interfaces

* Added a ``PerturbImage`` interface for taking an image stimulus and
  generating a perturbed image.

* Added a ``ScoreDetection`` interface that takes in ground-truth and predicted
  BBox-label pairs and generates scores based on a given metric.

Implementations

* Added several ``PerturbImage`` implementations:

  * From ``opencv``:

    * ``AverageBlurPerturber``: Applies average blurring to the given image
      stimulus.

    * ``GaussianBlurPerturber``: Applies Gaussian blurring to the given image
      stimulus.

    * ``MedianBlurPerturber``: Applies median blurring to the given image
      stimulus.

  * From ``skimage``:

    * ``SaltNoisePerturber``, ``PepperNoisePerturber``,
      ``SaltAndPepperNoisePerturber``: Adds salt and/or pepper noise to given
      image stimulus.

    * ``GaussianNoisePerturber``: Adds Gaussian-distributed additive noise to
      given image stimulus.

    * ``SpeckleNoisePerturber``: Adds multiplicative (Gaussian) noise to given
      image stimulus.

  * ``NOPPerturber``: Serves as a pass-through NOP perturber to test interface
    functionality.

* Added a ``ScoreDetection`` implementation, ``NOPScorer`` which serves
  as a pass-through NOP scorer to test interface functionality.

Fixes
-----
