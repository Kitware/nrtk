Pending Release Notes
=====================

Updates / New Features
----------------------

Examples

* Added an example notebook exploring the current state of several augmentation
  tools as well as the usability of the JATIC Toolbox augmentation protocol.

* Added an example notebook demonstrating NRTK perturber functionality.

* Added an example notebook demonstrating NRTK generator functionality.

Interfaces

* Added a ``GenerateObjectDetectorBlackboxResponse`` interface for generating
  response curves with given perturber factories, detector, and scorer.

* Added a ``PerturbImage`` interface for taking an image stimulus and
  generating a perturbed image.

* Added a ``PerturbImageFactory`` interface for generating ``PerturbImage``
  instances of specified type and configuration while varying one parameter.

* Added a ``ScoreDetection`` interface that takes in ground-truth and predicted
  BBox-label pairs and generates scores based on a given metric.

Implementations

* Add an example ``GenerateObjectDetectorBlackboxResponse`` implementation,
  ``SimpleGenerator`` which takes takes input data directly as Sequences.

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

  * From ``PIL``:

    * ``BrightnessPerturber``: Adjusts given image stimulus brightness.

    * ``ColorPerturber``: Adjusts given image stimulus color balance.

    * ``ContrastPerturber``: Adjust given image stimulus contrast.

    * ``SharpnessPerturber``: Adjust given image stimulus sharpness.

  * ``NOPPerturber``: Serves as a pass-through NOP perturber to test interface
    functionality.

* Added a ``PerturbImageFactory`` implementation, ``StepPerturbImageFactory``,
  which is a simple implementation that varies a chosen parameter from
  ``start`` to ``stop`` by the given ``step`` value.

* Added ``ScoreDetection`` implementations

  * ``NOPScorer``: Serves as a pass-through NOP scorer to test interface
    functionality.

  * ``RandomScorer``: Generates random score values and serves as a test for
    reproducibility.

  * ``COCOScorer``: Generates detection scores for a specific statistic index
    using the converted COCO format data.

  * ``ClassAgnosticPixelwiseIoUScorer``: Generates pixelwise IoU scores in a
    class agnostic way.

Fixes
-----
