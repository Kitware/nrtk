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


Fixes
-----
